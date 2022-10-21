#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.parallel
# import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import multiprocessing
import math
from tqdm.auto import tqdm
from dotenv import load_dotenv

load_dotenv()

# Carrega data path da pasta atual ou caminho predefinido em
# .env:ABS_PATH
try:
    DATA_PATH = os.path.abspath(__file__).dirname().join("data")
except Exception:
    DATA_PATH = os.environ.get("ABS_PATH")
# Número de thread worker para o dataloader
# (75% das threads disponíveis)
WORKERS = math.floor(multiprocessing.cpu_count()*0.75)
# Tamanho da amostra por iteração
BATCH_SIZE = 128
# Tamanho da imagem redimensionada
IMG_SIZE = 64
# Número de canais da imagem (3 - RGB)
NC = 3
# Tamanho do vertor latente de entrada
NZ = 100
# Número de canais para gerar camadas nas redes
NGF = NDF = IMG_SIZE
# Iterações de treinamento
EPOCHS = 5
# Learning rate para otimizadores
LEARNING_RATE = 2e-4
# 'Beta' dos otimizadores Adam
BETA = (0.5, 0.999)
# Slope da reta negativa do LeakyReLU
RELU_SLOPE = 0.2

# Labels para batches
LREAL = 1
LFAKE = 0

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):
    def __init__(self):
        # Inicializa módulo
        super(Generator, self).__init__()
        # Transformações convolucionais sequênciais sobre a entrada
        # (vide `../refs/CamadasGerador.png`)
        self.main = nn.Sequential(
            # Primeira camada convolucional da entrada
            # - Recebe uma distribuição normal de tamanho (100,)
            #   para sampling e transpões convolução em uma
            #   saída de TAMANHO DE IMAGEM x 2^3 canais
            nn.utils.spectral_norm(
                nn.ConvTranspose2d(
                    NZ, NGF*(2**3),
                    4, 1, 0,
                    bias=False
                )
            ),
            # nn.ConvTranspose2d(NZ, NGF*(2**3), 4, 1, 0, bias=False),
            nn.BatchNorm2d(NGF*(2**3)),
            # nn.LeakyReLU(RELU_SLOPE, inplace=True),
            nn.ReLU(inplace=True),
            # Segunda camada convolucional da entrada
            # - Recebe uma saída da camada anterior e transpõe
            #   com convolução em uma saída de TAMANHO DE IMAGEM x 2^2 canais
            nn.utils.spectral_norm(
                nn.ConvTranspose2d(
                    NGF*(2**3), NGF*(2**2),
                    4, 2, 1,
                    bias=False
                )
            ),
            # nn.ConvTranspose2d(NGF*(2**3), NGF*(2**2), 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF*(2**2)),
            # nn.LeakyReLU(RELU_SLOPE, inplace=True),
            nn.ReLU(inplace=True),
            # Terceira camada convolucional da entrada
            # - Recebe uma saída da camada anterior e transpõe
            #   com convolução em uma saída de TAMANHO DE IMAGEM x 2^1 canais
            nn.utils.spectral_norm(
                nn.ConvTranspose2d(
                    NGF*(2**2), NGF*(2**1),
                    4, 2, 1,
                    bias=False
                )
            ),
            # nn.ConvTranspose2d(NGF*(2**2), NGF*(2**1), 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF*(2**1)),
            # nn.LeakyReLU(RELU_SLOPE, inplace=True),
            nn.ReLU(inplace=True),
            # Quarta camada convolucional da entrada
            # - Recebe uma saída da camada anterior e transpõe
            #   com convolução em uma saída de TAMANHO DE IMAGEM x 2^0 canais
            nn.utils.spectral_norm(
                nn.ConvTranspose2d(
                    NGF*(2**1), NGF,
                    4, 2, 1,
                    bias=False
                )
            ),
            # nn.ConvTranspose2d(NGF*(2**1), NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF),
            # nn.LeakyReLU(RELU_SLOPE, inplace=True),
            nn.ReLU(inplace=True),
            # Camada final de convolução com saída de 3 canais (RGB),
            # com valores [ -1, 1 ] (ativação tanh)
            nn.utils.spectral_norm(
                nn.ConvTranspose2d(
                    NGF, NC,
                    4, 2, 1,
                    bias=False
                )
            ),
            # nn.ConvTranspose2d(NGF, NC, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: np.ndarray):
        return self.main(z)


class Discriminator(nn.Module):
    def __init__(self):
        # Inicializa módulo
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Primeira camada convolucional da entrada
            # - Recebe uma imagem de 3 canais (RGB) e transpõe
            #   com convolução em uma saída de TAMANHO DE IMAGEM canais
            nn.utils.spectral_norm(nn.Conv2d(NC, NDF, 4, 2, 1, bias=False)),
            # nn.Conv2d(NC, NDF, 4, 2, 1, bias=False),
            nn.LeakyReLU(RELU_SLOPE, inplace=True),
            # Segunda camada convolucional da entrada
            # - Recebe uma saída da camada anterior e transpõe
            #   com convolução em uma saída de TAMANHO DE IMAGEM x 2^1 canais
            nn.utils.spectral_norm(nn.Conv2d(NDF, NDF*(2**1), 4, 2, 1, bias=False)),
            # nn.Conv2d(NDF, NDF*(2**1), 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF*(2**1)),
            nn.LeakyReLU(RELU_SLOPE, inplace=True),
            # Terceira camada convolucional da entrada
            # - Recebe uma saída da camada anterior e transpõe
            #   com convolução em uma saída de TAMANHO DE IMAGEM x 2^2 canais
            nn.utils.spectral_norm(nn.Conv2d(NDF*(2**1), NDF*(2**2), 4, 2, 1, bias=False)),
            # nn.Conv2d(NDF*(2**1), NDF*(2**2), 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF*(2**2)),
            nn.LeakyReLU(RELU_SLOPE, inplace=True),
            # Quarta camada convolucional da entrada
            # - Recebe uma saída da camada anterior e transpõe
            #   com convolução em uma saída de TAMANHO DE IMAGEM x 2^3 canais
            nn.utils.spectral_norm(nn.Conv2d(NDF*(2**2), NDF*(2**3), 4, 2, 1, bias=False)),
            # nn.Conv2d(NDF*(2**2), NDF*(2**3), 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF*(2**3)),
            nn.LeakyReLU(RELU_SLOPE, inplace=True),
            # Camada final de convolução com saída de 1 canal
            # (classificador binário, one-hot-encoding) de ativação
            # Sigmoid
            nn.utils.spectral_norm(nn.Conv2d(NDF*(2**3), 1, 4, 1, 0, bias=False)),
            # nn.Conv2d(NDF*(2**3), 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, z: np.ndarray):
        return self.main(z)


# Cria dataset da pasta
dataset = dsets.ImageFolder(
    root=DATA_PATH,
    transform=transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])
)

# Carregador do dataset em batches
dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=WORKERS
)

# Exibe imagens originais
plt.imshow(
    np.transpose(
        vutils.make_grid(
            next(iter(dataloader))[0].to(DEVICE)[:64],
            padding=2,
            normalize=True
        ).cpu(),
        (1, 2, 0)
    )
)
plt.title("Imagens originais pré-processadas")
plt.show()


# Inicializa os pesos do módulo dado com uma
# distruição normal (média: 0, variância: 0.2)
def initWeights(module: nn.Module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(module.weight.data, 0., 0.2)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1., 0.2)
        nn.init.constant_(module.bias.data, 0)


# Cria rede geradora com pesos normais
netG = Generator().to(DEVICE)
if DEVICE.type == "cuda":
    netG = nn.DataParallel(netG, list(range(torch.cuda.device_count())))
netG.apply(initWeights)
print(netG)

# Cria rede discriminadora com pesos normais
netD = Discriminator().to(DEVICE)
if DEVICE.type == "cuda":
    netD = nn.DataParallel(netD, list(range(torch.cuda.device_count())))
netD.apply(initWeights)
print(netD)

# Função de perda
criterion = nn.BCELoss()

# Seed randômica normal para gerar imagens para exibição posteriormente
seed = torch.randn(64, NZ, 1, 1, device=DEVICE)

# Otimizadores das redes
optimizerD = optim.Adam(
    netD.parameters(),
    lr=LEARNING_RATE*0.1,  # DEBUG
    betas=BETA
)
optimizerG = optim.Adam(
    netG.parameters(),
    lr=LEARNING_RATE,
    betas=BETA
)

# schedulingMilestones = [
#     math.floor(EPOCHS*0.25),
#     math.floor(EPOCHS*0.5),
#     math.floor(EPOCHS*0.75)
# ]
# schedulerD = optim.lr_scheduler.MultiStepLR(
#     optimizerD,
#     milestones=[
#         math.floor(EPOCHS*0.33),
#         math.floor(EPOCHS*0.66)
#     ]
# )
# schedulerG = optim.lr_scheduler.MultiStepLR(
#     optimizerG,
#     milestones=[
#         math.floor(EPOCHS*0.25),
#         math.floor(EPOCHS*0.5),
#         math.floor(EPOCHS*0.75)
#     ]
# )

print("\nStarting training...")

# Perdas de ambas as redes para grafar ao longo do tempo
G_losses = []
D_losses = []

print("Running epochs...")
# Para cada iteração de treinamento
for epoch in range(EPOCHS):
    progressbar = tqdm(enumerate(dataloader, 0), total=len(dataloader))
    progressbar.set_description("Epoch [{0}/{1}]".format(epoch + 1, EPOCHS))
    # Para cada batch do dataloader
    for i, batch in progressbar:
        # --------------------------------------------------------------
        # Treina discriminador, maximizando log(D(x)) + log(1 - D(G(z)))
        # --------------------------------------------------------------

        # Treinamento com batch real

        # Zera gradientes dos pesos do discriminador
        netD.zero_grad()
        # Aloca batch no contexto
        real = batch[0].to(DEVICE)
        realSize = real.size(0)
        labels = torch.full(
            size=(realSize, ),
            fill_value=LREAL,
            dtype=torch.float,
            device=DEVICE
        )
        # Saída linearizada com entrada real
        output = netD(real).view(-1)
        # Erro do conjunto real
        errD_Real = criterion(output, labels)
        # Gradientes dado erro
        errD_Real.backward()
        D_x = output.mean().item()

        # Treinamento com batch falsa

        # Gera z gaussiano
        noise = torch.randn(realSize, NZ, 1, 1, device=DEVICE)
        # Gera saída falsa
        fake = netG(noise)
        # Recria labels para discriminar falso
        labels.fill_(LFAKE)
        # Classificação das imagens fake geradas
        # (detach -> retira saída do contexto do gerador, para só atualizar
        # os pesos do discriminador)
        output = netD(fake.detach()).view(-1)
        # Erro do conjunto falso
        errD_Fake = criterion(output, labels)
        # Gradientes dado erro
        errD_Fake.backward()
        # Média dos valores discriminados do gerador
        D_G_z1 = output.mean().item()
        # Erro do discriminador como soma do erro com
        # imagens falsas e reais
        errD = errD_Fake + errD_Real
        # Atualiza os pesos do discriminador
        optimizerD.step()

        # --------------------------------------------------------------
        # Treina gerador, maximizando log(D(G(z)))
        # --------------------------------------------------------------

        # Zera gradientes dos pesos do discriminador
        netG.zero_grad()
        # Labels como real, que seria a geração ideal
        labels.fill_(LREAL)
        # Saída da discriminação da saída falsa gerada no último passo
        output = netD(fake).view(-1)
        # Perda do gerador
        errG = criterion(output, labels)
        # Gradientes dado erro
        errG.backward()
        # Média dos valores discriminados do gerador após otimização
        # do discriminador
        D_G_z2 = output.mean().item()
        # Atualiza os pesos do discriminador
        optimizerG.step()

        # Stats a cada 5 batches
        if i % 10 == 0:
            print(
                '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): '
                '%.4f\tD(G(z)): %.4f / %.4f' % (
                    epoch, EPOCHS,
                    i, len(dataloader),
                    errD.item(), errG.item(),
                    D_x, D_G_z1, D_G_z2
                ))

        # Salvas perdas dessa iteração/batch
        G_losses.append(errG.item())
        D_losses.append(errD.item())
    # DEBUG
    # schedulerD.step()
    # schedulerG.step()

plt.title("Perdas ao longo do tempo")
plt.plot(G_losses, label="Perda do gerador")
plt.plot(D_losses, label="Perda do discriminador")
plt.xlabel("Iterações")
plt.ylabel("Perda")
plt.legend()
plt.show()

with torch.no_grad():
    # Gera imagens com seed
    fake = netG(seed).detach().cpu()
    # Exibe imagens geradas
    plt.imshow(
        np.transpose(
            vutils.make_grid(
                fake,
                padding=2,
                normalize=True
            ).cpu(),
            (1, 2, 0)
        )
    )
    plt.title("Imagens geradas")
    plt.show()
