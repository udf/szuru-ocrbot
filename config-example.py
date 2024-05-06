BOORU_URL = 'https://booru.example.com/'
API_BASE_URL = f'{BOORU_URL}api/'
USERNAME = 'ocrbot'
# token created from the user page
TOKEN = 'abcdef12-3456-7890-abcd-ef1234567890'

# tag for posts that we should run on
TAG_PENDING = 'pls_ocr_thx'
# tag for posts that we have run on
TAG_DONE = 'ocr'
# tag for posts that we encountered an error on
TAG_ERROR = 'ocr_error'

# matches below this score are discarded
SCORE_THRESHOLD = 0.25
# matches above or equal to this score will be merged with neighbouring matches
MERGE_SCORE_THRESHOLD = 0.5
# merge matches whose bounding boxes are at most this far from each other
BOX_MERGE_THRESHOLD = 0.05