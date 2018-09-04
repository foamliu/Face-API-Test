import cognitive_face as CF

KEY = 'subscription key'  # Replace with a valid Subscription Key here.
CF.Key.set(KEY)

BASE_URL = 'http://localhost:5000/face/v1.0/'  # Replace with your regional Base URL
CF.BaseUrl.set(BASE_URL)

img_url = 'https://raw.githubusercontent.com/Microsoft/Cognitive-Face-Windows/master/Data/detection1.jpg'
result = CF.face.detect(img_url)
print(result)