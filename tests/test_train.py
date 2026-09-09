from src.train import train 

def test_accuracy():
    acc = train(test_size=0.2, random_state=42)
    assert acc >= 0.9