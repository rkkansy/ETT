import os
import sys
from datasets import load_dataset

def download_bookcorpus(book_path):
    os.makedirs(book_path, exist_ok=True)

    try:
        if not os.path.isfile(os.path.join(book_path, 'bookcorpus_train.txt')):
            dataset = load_dataset("bookcorpus", split='train')
            output_file = os.path.join(book_path, 'bookcorpus_train.txt')
            print("Preprocessing bookcorpus.")
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in dataset:
                    f.write(item['text'] + '\n')

            print(f"Downloaded and saved BookCorpus data to {output_file}")
        else:
            print("Dataset already downloaded.")
    except Exception as e:
        print(f"Failed to download train split: {e}")

if __name__ == "__main__":
    book_path = sys.argv[1]
    download_bookcorpus(book_path)