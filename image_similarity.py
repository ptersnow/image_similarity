import os
import glob
import json

from PIL import Image
from sentence_transformers import SentenceTransformer, util


Image.MAX_IMAGE_PIXELS = None


print("="*80)
print('Carregando CLIP Model...')
model = SentenceTransformer('clip-ViT-L-14')
print("="*80)
print()

novos_pronacs = ['C']

working_directory = os.path.dirname(os.path.realpath(__file__))
images_dir = os.path.join(working_directory, "pronacs_teste")
results_dir = os.path.join(working_directory, "analise")

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

def load_images_from_directory(directory):
    image_paths = []
    for ext in ('**/*.jpg', '**/*.jpeg', '**/*.png', '**/*.gif'):
        image_paths.extend(glob.glob(os.path.join(directory, ext), recursive=True))
    return image_paths


def find_duplicates_and_similar_images(pronac, image_paths1, image_paths2):
    duplicates = []
    threshold = 0.75

    encoded_images1 = model.encode([Image.open(filepath) for filepath in image_paths1], batch_size=128, convert_to_tensor=True)
    encoded_images2 = model.encode([Image.open(filepath) for filepath in image_paths2], batch_size=128, convert_to_tensor=True)

    results = util.cos_sim(encoded_images1, encoded_images2)

    for i in range(len(encoded_images1)):
        for j in range(len(encoded_images2)):
            if (results[i, j] >= threshold):
                duplicates.append({
                                  "pronac": pronac,
                                  "image1": image_paths1[i],
                                  "image2": image_paths2[j],
                                  "score": results[i, j].item()
                                  })
    return duplicates


if __name__ == "__main__":

    data = []
    for i, novo_pronac in enumerate(novos_pronacs):
        out_file = os.path.join(results_dir, novo_pronac + ".json")

        image_paths1 = load_images_from_directory(os.path.join(images_dir, novo_pronac))
        print("="*30, novo_pronac, "="*30)
        print(f"Encontrado {len(image_paths1)} imagens em {novo_pronac}")

        if len(image_paths1) > 0:

            if len(novos_pronacs) > 1:
                print("Comparando com novos pronacs")
                for j in range(i + 1, len(novos_pronacs)):
                    image_paths2 = load_images_from_directory(novos_pronacs[j])
            
                    print("-"*80)
                    print(f"Encontrado {len(image_paths2)} imagens em {novos_pronacs[j]}")

                    if len(image_paths2) > 0:
                        print(f'Buscando duplicatas entre {novo_pronac} e {novos_pronacs[j]}...')
                        data.extend(find_duplicates_and_similar_images(novos_pronacs[j], image_paths1, image_paths2))

            print("Comparando com pronacs antigos")
            with os.scandir(images_dir) as it:
                for entry in it:
                    if not entry.name in novos_pronacs:
                        image_paths2 = load_images_from_directory(entry)
        
                        print("-"*80)
                        print(f"Encontrado {len(image_paths2)} imagens em {entry.name}")

                        if len(image_paths2) > 0:
                            print(f'Buscando duplicatas entre {novo_pronac} e {entry.name}...')
                            data.extend(find_duplicates_and_similar_images(entry.name, image_paths1, image_paths2))

            print("-"*80)
            print(f"Salvando arquivo {out_file}... ", end="")
            with open(out_file, 'w') as fp:
                json.dump(data, fp, indent=4)
            print("[DONE]")
        else:
            print("Nada a fazer")