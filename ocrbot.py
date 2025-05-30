import logging
import math
import os
import sys
import urllib.parse
from base64 import urlsafe_b64encode
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from typing import List, Tuple

from requests import exceptions
from requests_toolbelt import sessions

import config
from config import BOX_MERGE_THRESHOLD, MERGE_SCORE_THRESHOLD, SCORE_THRESHOLD


os.nice(10)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ocrbot')
clamp = lambda val, low, high: min(max(low, val), high)
ocr_reader = None


class Position(Enum):
  LEFT = 1
  RIGHT = 2
  TOP = 3
  BOTTOM = 4


@dataclass
class Box:
  left: float
  right: float
  top: float
  bottom: float

  @staticmethod
  def from_xy(xy1, xy2):
    return Box(
      min(xy1[0], xy2[0]),
      max(xy1[0], xy2[0]),
      min(xy1[1], xy2[1]),
      max(xy1[1], xy2[1]),
    )

  def map(self, func):
    return Box(
      func(self.left),
      func(self.right),
      func(self.top),
      func(self.bottom)
    )

  def merge(self, other: 'Box'):
    return Box(
      min(self.left, other.left),
      max(self.right, other.right),
      min(self.top, other.top),
      max(self.bottom, other.bottom),
    )

  def as_poly(self):
    return [
      [self.left, self.top],
      [self.right, self.top],
      [self.right, self.bottom],
      [self.left, self.bottom],
    ]

  def intersects(self, other, margin=0.0):
    return not (
      self.right + margin < other.left
      or self.left - margin > other.right
      or self.bottom + margin < other.top
      or self.top - margin > other.bottom
    )

  def get_center(self):
    return (self.left + self.right) / 2, (self.top + self.bottom) / 2

  def get_area(self):
    return abs(self.left - self.right) * abs(self.top - self.bottom) 

  def clip_x(self, low, high):
    return Box(clamp(self.left, low, high), clamp(self.right, low, high), self.top, self.bottom)

  def clip_y(self, low, high):
    return Box(self.left, self.right, clamp(self.top, low, high), clamp(self.bottom, low, high))

  def find_nearest_side(self, other: 'Box') -> Tuple[float, Position]:
    center_x, center_y = self.get_center()
    areas = [
      (other.clip_x(0, self.left).get_area(), max(0, self.left - other.right), Position.LEFT),
      (other.clip_x(self.right, 1).get_area(), max(0, other.left - self.right), Position.RIGHT),
      (other.clip_y(0, self.top).get_area(), max(0, self.top - other.bottom), Position.TOP),
      (other.clip_y(self.bottom, 1).get_area(), max(0, other.top - self.bottom), Position.BOTTOM),
      # measure by the area on either side of the center
      # (in the case of fully contained boxes, the above areas will be 0)
      (other.clip_x(0, center_x).get_area(), 0, Position.LEFT),
      (other.clip_x(center_x, 1).get_area(), 0, Position.RIGHT),
      (other.clip_y(0, center_y).get_area(), 0, Position.TOP),
      (other.clip_y(center_y, 1).get_area(), 0, Position.BOTTOM),
    ]

    max_area, dist, position = max(areas, key=lambda v: v[0])
    return dist, position


@dataclass
class TextBox:
  box: Box
  text: str
  score: float


def merge_close_boxes(boxes: List[TextBox]) -> List[TextBox]:
  i = 0
  while i < len(boxes):
    cur = boxes[i]
    for j in range(i + 1, len(boxes)):
      other = boxes[j]
      if not cur.box.intersects(other.box, margin=BOX_MERGE_THRESHOLD):
        continue
      dist, position = cur.box.find_nearest_side(other.box)
      if dist > BOX_MERGE_THRESHOLD:
        continue

      match position:
        case Position.TOP:
          cur.text = f'{other.text}\n{cur.text}'
        case Position.BOTTOM:
          cur.text = f'{cur.text}\n{other.text}'
        case Position.LEFT:
          cur.text = f'{other.text} {cur.text}'
        case Position.RIGHT:
          cur.text = f'{cur.text} {other.text}'
      cur.box = cur.box.merge(other.box)
      cur.score = (cur.score + other.score) / 2
      boxes.pop(j)
      break
    else:
      # nothing to merge, go to next box
      i += 1
      continue

  return boxes


def do_ocr(image_handle) -> Tuple[List[TextBox], List[TextBox]]:
  import cv2
  import easyocr
  import numpy as np
  from PIL import Image
  global ocr_reader
  if not ocr_reader:
    ocr_reader = easyocr.Reader(["en"])

  # alpha blend a white background to fix opencv not checking the alpha
  im = Image.open(image_handle).convert('RGBA')
  new_im = Image.new('RGBA', im.size, 'white')
  new_im.paste(im, mask=im)
  im = None
  new_im = new_im.convert('RGB')
  new_im.thumbnail((2048, 2048), resample=Image.Resampling.LANCZOS)
  new_im_handle = BytesIO()
  new_im.save(new_im_handle, format='png', compress_level=1)
  new_im = None

  img = cv2.imdecode(np.frombuffer(new_im_handle.getbuffer(), np.uint8), cv2.IMREAD_COLOR)
  height, width, _ = img.shape
  boxes: List[TextBox] = []
  low_score_boxes: List[TextBox] = []
  for bbox, text, score in ocr_reader.readtext(img, low_text=0.3):
    if score < SCORE_THRESHOLD:
      continue
    bbox = [[p[0] / width, p[1] / height] for p in bbox]
    box = TextBox(Box.from_xy(bbox[0], bbox[2]), text, score)
    if score >= MERGE_SCORE_THRESHOLD:
      boxes.append(box)
    else:
      box.text = f'(? {round(score * 100)}%) {box.text}'
      low_score_boxes.append(box)

  return merge_close_boxes(boxes), low_score_boxes


def process_post(post_id, url):
  logger.info(f'#{post_id}: Downloading {url}')
  url = urllib.parse.urljoin(config.BOORU_URL, post['contentUrl'])
  image = BytesIO()
  with session.get(url, stream=True) as r:
    for chunk in r.iter_content(chunk_size=1024*1024):
      image.write(chunk)
  image.seek(0)
  logger.info(f'#{post_id}: Processing')
  high_score_boxes, low_score_boxes = do_ocr(image)
  logger.info(f'#{post_id} Found {len(high_score_boxes)}+{len(low_score_boxes)} texts')
  return high_score_boxes + low_score_boxes


def update_post(session, post_id, remove_tags=set(), add_tags=set(), notes=None):
  logger.info(f'#{post_id} Updating: {len(notes or [])} notes (+{",".join(add_tags)} -{",".join(remove_tags)})')
  post = session.get(
    f'post/{post_id}',
    params={
      'fields': 'version,tags'
    },
    timeout=5
  ).json()
  new_data = {
    'version': post['version'],
    'tags': list(({tag['names'][0] for tag in post['tags']} | add_tags) - remove_tags),
  }
  if notes:
    new_data['notes'] = notes
  session.put(f'post/{post_id}', json=new_data, timeout=5)


if __name__ == '__main__':
  if len(sys.argv) > 1:
    for filename in sys.argv[1:]:
      with open(filename, 'rb') as f:
        b1, b2 = do_ocr(f)
      for b in b1:
        b.box = b.box.map(lambda n: round(n, 2))
        print(b)
      for b in b2:
        print(b)
    exit(0)

  session = sessions.BaseUrlSession(base_url=config.API_BASE_URL)
  encoded_token = urlsafe_b64encode(f"{config.USERNAME}:{config.TOKEN}".encode('ascii')).decode('ascii')
  session.headers = {
    'Accept': 'application/json',
    'Content-Type': 'application/json',
    'Authorization': f'Token {encoded_token}'
  }
  session.hooks = {
    'response': lambda r, *args, **kwargs: r.raise_for_status()
  }

  logger.info('Fetching posts...')
  try:
    posts = session.get(
      'posts',
      params={
        'query': f'type:image {config.TAG_PENDING}',
        'limit': 10,
        'fields': 'id,contentUrl'
      },
      timeout=5
    ).json()
  except exceptions.HTTPError:
    logger.exception('Error connecting to server')
    exit()

  logger.info(f'Fetched {len(posts["results"])} post(s)!')
  for post in posts['results']:
    post_id = post['id']
    notes = None
    try:
      textboxes = process_post(post_id, post['contentUrl'])
      notes = [
        {
          'polygon': textbox.box.map(lambda n: min(max(0, n), 1)).as_poly(),
          'text': textbox.text
        }
        for textbox in textboxes
      ]
    except Exception as e:
      logger.exception(f'#{post_id}: Error running OCR')
    update_post(
      session,
      post_id,
      remove_tags={config.TAG_ERROR if notes else config.TAG_DONE, config.TAG_PENDING},
      add_tags={config.TAG_DONE if notes else config.TAG_ERROR},
      notes=notes
    )
