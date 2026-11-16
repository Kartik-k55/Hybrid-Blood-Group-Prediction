import urllib.request
import json

def test_api():
    res = urllib.request.urlopen('http://127.0.0.1:5000/api/samples')
    data = json.loads(res.read().decode())
    samples = data.get('samples', [])
    print(f"Total samples found: {len(samples)}")
    
    if samples:
        sample_path = samples[0]['path']
        print(f"Testing prediction on sample: {sample_path}")
        req = urllib.request.Request(
            'http://127.0.0.1:5000/api/predict',
            data=json.dumps({'sample_path': sample_path, 'model_type': 'vgg16_hybrid'}).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode())
            print("API Response:")
            print(json.dumps(result, indent=2))

if __name__ == '__main__':
    test_api()
