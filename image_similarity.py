import os
import glob
import json
import fitz
import pdfkit
import jinja2
import shutil

from PIL import Image
from datetime import date
from sentence_transformers import SentenceTransformer, util


Image.MAX_IMAGE_PIXELS = None

threshold = 0.75

print("="*80)
print('Carregando CLIP Model...')
model = SentenceTransformer('clip-ViT-L-14')
print("="*80)
print()

working_directory = os.path.dirname(os.path.realpath(__file__))
sources_dir = os.path.join(working_directory, "novos")
reports_dir = os.path.join(working_directory, "analise")
embeddings_dir = os.path.join(working_directory, "embeddings")
verified_dir = os.path.join(working_directory, "verificados")


def get_date() -> str:
    """
    Get current date in format dd/mm/yyyy

    :return: current date
    """
    today = date.today()
    return today.strftime("%d/%m/%Y")


def load_images_from_directory(directory: str) -> list:
    """
    Load images from directory

    :param directory: directory path
    
    :return: list of image paths
    """
    image_paths = []

    dimlimit = 100  # each image side must be greater than this
    relsize = 0.05  # image size ratio must be larger than this (5%)
    abssize = 2048  # absolute image size limit 2 KB: ignore if smaller

    ext_pdfs = ('**/*.pdf', '**/*.PDF')
    ext_images = ('**/*.jpg', '**/*.JPG', '**/*.jpeg', '**/*.JPEG', '**/*.png', '**/*.PNG')

    for ext in ext_pdfs:
        pdfs = glob.glob(os.path.join(directory, ext), recursive=True)
        
        for pdf in pdfs:
            doc = fitz.open(pdf) # open the document
            for page_index in range(len(doc)):
                page = doc[page_index] # load the page
                image_list = page.get_images() # get list of images on the page

                file_name = os.path.splitext(pdf)[0]

                for image_index, image in enumerate(image_list, start=1): # enumerate the image list
                    xref = image[0] # get the XREF of the image                    
                    width = image[2]
                    height = image[3]

                    if min(width, height) <= dimlimit:
                        continue

                    pix = fitz.Pixmap(doc, xref) # create a Pixmap

                    if pix.n - pix.alpha > 3: # CMYK: convert to RGB first
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    imgdata = pix.tobytes("png") # convert to PNG image bytes

                    if len(imgdata) <= abssize:
                        continue

                    if len(imgdata) / (width * height * (pix.n - pix.alpha)) <= relsize:
                        continue

                    pix.save(f"{file_name}_PDF_pagina_{page_index + 1}_figura_{image_index}.png") # save the image as png
                    pix = None
                    
    for ext in ext_images:
        image_paths.extend(glob.glob(os.path.join(directory, ext), recursive=True))
    return image_paths


def generateEmbeddings(pronac: str, image_paths: list) -> list:
    """
    Generate embeddings for images in image_paths

    :param pronac: pronac number
    :param image_paths: list of image paths

    :return: list of embeddings
    """
    embeddings = []
    pronac_file = os.path.join(embeddings_dir, f"{pronac}.json")

    if not os.path.exists(pronac_file):
        print(f"Gerando embeddings para {pronac}... ", end="")
        try:
            embeddings = model.encode([Image.open(filepath) for filepath in image_paths], batch_size=128, convert_to_tensor=True)

            json_data = {
                "images": image_paths,
                "embeddings": embeddings.tolist()
            }

            with open(pronac_file, 'w') as fp:
                json.dump(json_data, fp, indent=4)

            print("[DONE]")
        except Exception as e:
            print("[ERROR]", e)
    else:
        json_data = json.load(open(pronac_file, "r"))
        embeddings = json_data["embeddings"]
    
    return embeddings



def render_html(pronac: str, analisados: list, similares: list, out_file: str) -> None:
    """
    Render html page using jinja based on templates/relatorio.html.jinja

    :param pronac: pronac number
    :param analisados: list of analyzed pronacs
    :param similares: list of similar pronacs
    :param out_file: output file name
    """
    template_file = "relatorio.html.jinja"
    template_loader = jinja2.FileSystemLoader(searchpath="./templates")
    template_env = jinja2.Environment(loader=template_loader)
    
    template = template_env.get_template(template_file)
    output_text = template.render(
        pronac=pronac,
        date=get_date(),
        analisados=analisados,
        similares=similares
    )

    print(f"Salvando arquivo {out_file}.html ... ", end="")
    with open(out_file + ".html", "w+") as html_file:
        html_file.write(output_text)
    print("[DONE]")

    print(f"Salvando arquivo {out_file}.pdf ... ", end="")
    html2pdf(out_file + ".html", out_file + ".pdf")
    print("[DONE]")


def html2pdf(html_path: str, pdf_path: str) -> None:
    """
    Convert html to pdf

    :param html_path: path to html file
    :param pdf_path: path to pdf file
    """

    options = {
        'page-size': 'A4',
        'margin-top': '2cm',
        'margin-right': '2cm',
        'margin-bottom': '2cm',
        'margin-left': '2cm',
        'encoding': "UTF-8",
        'enable-local-file-access': ''
    }

    with open(html_path) as f:
        pdfkit.from_file(f, pdf_path, options=options, verbose=True)


def find_similarities() -> None:
    """
    Find similarities between images in sources_dir and verified_dir
    """
    with os.scandir(sources_dir) as source_iterator:
        for novo_pronac in source_iterator:
            similares = []
            analisados = []

            source_dir = os.path.join(sources_dir, novo_pronac)
            out_file = os.path.join(reports_dir, novo_pronac.name)

            image_paths1 = load_images_from_directory(source_dir)
            print("="*30, novo_pronac.name, "="*30)
            print(f"Encontrado {len(image_paths1)} imagens em {novo_pronac.name}")

            if len(image_paths1) > 0:
                new_embeddings = generateEmbeddings(novo_pronac.name, image_paths1)

                print("Comparando com pronacs antigos")
                with os.scandir(verified_dir) as verified_iterator:
                    for velho_pronac in verified_iterator:
                        image_paths2 = load_images_from_directory(velho_pronac)
        
                        print("-"*80)
                        print(f"Encontrado {len(image_paths2)} imagens em {velho_pronac.name}")

                        similaridades = []
                        if len(image_paths2) > 0:
                            old_embeddings = generateEmbeddings(velho_pronac.name, image_paths2)

                            print(f'Buscando similaridades entre {novo_pronac.name} e {velho_pronac.name}...')
                            results = util.cos_sim(new_embeddings, old_embeddings)

                            for i in range(len(new_embeddings)):
                                for j in range(len(old_embeddings)):
                                    if (results[i, j] >= threshold):
                                        similaridades.append({
                                            "image1": image_paths1[i],
                                            "image2": image_paths2[j],
                                            "score": int(results[i, j].item() * 100)
                                        })

                            similares.append({
                                "pronac": velho_pronac.name,
                                "similaridades": similaridades
                            })

                        analisados.append({
                            "pronac": velho_pronac.name,
                            "arquivos": len(image_paths2),
                            "similaridades": len(similaridades)
                        })

                print("-"*80)
            else:
                print("Nada a fazer")

            # json_data = {
            #     "analisados": analisados,
            #     "similares": similares
            # }

            # print(f"Salvando arquivo {out_file}.json ... ", end="")
            # with open(out_file + '.json', 'w') as fp:
            #     json.dump(json_data, fp, indent=4)
            # print("[DONE]")

            render_html(novo_pronac.name, analisados, similares, out_file)

            print(f"Movendo {source_dir} para {verified_dir} ", end="")
            shutil.move(source_dir, verified_dir)
            print("[DONE]")


if __name__ == "__main__":

    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)

    if not os.path.exists(verified_dir):
        os.makedirs(verified_dir)

    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)

    find_similarities()
            
