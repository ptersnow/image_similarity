import os
import glob
import json
import jinja2
import pdfkit
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
verified_dir = os.path.join(working_directory, "verificados")


def get_date():
    today = date.today()
    return today.strftime("%d/%m/%Y")


def load_images_from_directory(directory):
    image_paths = []
    for ext in ('**/*.jpg', '**/*.jpeg', '**/*.png', '**/*.gif'):
        image_paths.extend(glob.glob(os.path.join(directory, ext), recursive=True))
    return image_paths


def render_html(pronac, analisados, similares, out_file):
    """
    Render html page using jinja based on templates/relatorio.html.jinja
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


def html2pdf(html_path, pdf_path):
    """
    Convert html to pdf using pdfkit which is a wrapper of wkhtmltopdf
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


def find_similarities():
    with os.scandir(sources_dir) as sit:
        for novo_pronac in enumerate(sit):
            similares = []
            analisados = []

            source_dir = os.path.join(sources_dir, novo_pronac)
            out_file = os.path.join(reports_dir, novo_pronac.name)

            image_paths1 = load_images_from_directory(source_dir)
            print("="*30, novo_pronac.name, "="*30)
            print(f"Encontrado {len(image_paths1)} imagens em {novo_pronac.name}")

            if len(image_paths1) > 0:

                new_embeds = model.encode([Image.open(filepath) for filepath in image_paths1], batch_size=128, convert_to_tensor=True)

                print("Comparando com pronacs antigos")
                with os.scandir(verified_dir) as it:
                    for velho_pronac in it:
                        image_paths2 = load_images_from_directory(velho_pronac)
        
                        print("-"*80)
                        print(f"Encontrado {len(image_paths2)} imagens em {velho_pronac.name}")

                        if len(image_paths2) > 0:
                            print(f'Buscando similaridades entre {novo_pronac.name} e {velho_pronac.name}...')
                            old_embeds = model.encode([Image.open(filepath) for filepath in image_paths2], batch_size=128, convert_to_tensor=True)

                            results = util.cos_sim(new_embeds, old_embeds)

                            similaridades = []
                            for i in range(len(new_embeds)):
                                for j in range(len(old_embeds)):
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

            json_data = {
                "analisados": analisados,
                "similares": similares
            }

            print(f"Salvando arquivo {out_file}.json ... ", end="")
            with open(out_file + '.json', 'w') as fp:
                json.dump(json_data, fp, indent=4)
            print("[DONE]")

            render_html(novo_pronac.name, analisados, similares, out_file)

            print(f"Movendo {source_dir} para {verified_dir} ", end="")
            shutil.move(source_dir, verified_dir)
            print("[DONE]")


if __name__ == "__main__":

    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)

    if not os.path.exists(verified_dir):
        os.makedirs(verified_dir)

    find_similarities()