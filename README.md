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

As bibliotecas do Pytorch devem ser instaladas manualmente, para evitar problemas com versão e com execução na GPU:

- Pytorch + CUDA 12.1

    ```
    pip3 install torch torchvision torchaudio
    ```

- Pytorch + CUDA 11.8

    ```
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

- Pytorch + CPU

    ```
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ```

A verificação de similaridades utiliza como base a pasta novos que estar no mesmo nível que o arquivo `image_similarity.py` e conter os pronacs que vão ser examinados. Em seguida, podemos executar o arquivo:

```
python3 image_similarity.py
```

Depois de ser verificada por similaridades, a pasta do pronac vai ser movida da pasta `novos` para a pasta `verificados`

### Resultados

O resultado das análises vai ser gerado em dois arquivos json e pdf dentro da pasta analise com o número do pronac como nome, seguido da extensão.
