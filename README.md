# Similaridade entre Imagens utilizando PyTorch

Esse projeto busca similaridades entre imagens dentro de pastas e subpastas

## Utilização

Primeiro, faça o clone do repositório:
```
git clone https://github.com/ptersnow/image_similarity.git
cd image_similarity
```

Para facilitar a utilização e atualização de pacotes, vamos utilizar o `virtualenv`:
```
python3 -m venv venv
source venv/bin/activate
```
Feito isso, podemos instalar as dependências:
```
pip install -r requirements.txt
```

Por hora, a verificação de similaridades utiliza como base o array novos_pronacs que deve ser alterado no arquivo `image_similarity.py`. Com os novos_pronacs no lugar, podemos executar o arquivo:
```
python3 image_similarity.py
```


### Dados

Uma pasta chamada pronacs deve ser criada com todos os pronacs dentro dela.

### Resultados

O resultado das análises vai ser gerado em um arquivo json dentro da pasta analise com o número do pronac como nome, seguido da extensão .json.
