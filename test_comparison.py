import urllib.request
import json

def test_comparison():
    req = urllib.request.Request(
        'http://127.0.0.1:5000/api/predict',
        data=json.dumps({'sample_path': 'sample dataset/sample_data.jpg', 'model_type': 'compare_all'}).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode())
        print("Comparison response received successfully:")
        data = res['data']
        print(f"Top / Recommended Prediction: {data['prediction']} ({data['confidence']}%)")
        print(f"Consensus Group: {data['consensus_group']}")
        print(f"Agreement: {data['agreement_count']} ({data['agreement_percentage']}%)")
        print("\nModel by Model Breakdown:")
        for m in data['models_comparison']:
            rec_tag = " [RECOMMENDED]" if m['is_recommended'] else ""
            print(f"  * {m['name']}{rec_tag}: {m['prediction']} ({m['confidence']}%) -> {m['accuracy_rank']}")

if __name__ == '__main__':
    test_comparison()
