# -*- coding: utf-8 -*-
"""Figuras do relatório (Figuras 1, 2, 3, 4 e A1). Entradas: CSVs e PNG gerados pelo notebook."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from PIL import Image

for nome in ("Liberation Sans", "Arial", "DejaVu Sans"):
    if any(f.name == nome for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = nome
        break

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
AZUL, LARANJA, AQUA, AMARELO, CINZA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"
MESES = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
plt.rcParams.update({"axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

def estilo(ax):
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(length=0)

def milhar(v, _=None):
    return f"{int(round(v)):,}".replace(",", ".")

# ------------------------------------------------------------ Figura 1 (série mensal, dois painéis, mesmo eixo de tempo)
r = pd.read_csv("resumo_mensal_multiano.csv")
anos = sorted(r.ano.unique()); ano_min = anos[0]
fig, (a, b) = plt.subplots(2, 1, figsize=(6.3, 4.8), sharex=True)
for ax in (a, b):
    estilo(ax)
    for an in anos:                                              # estação seca (jul–nov) de cada ano
        ini = (an - ano_min) * 12
        ax.axvspan(ini + 6.5, ini + 11.5, color="#efeeea", zorder=0)
a.plot(r.periodo, r.qtd_focos, color=LARANJA, lw=1.8, marker="o", ms=3.5, mec=SURFACE, mew=0.8, zorder=3)
b.plot(r.periodo, r.qtd_internacoes, color=AZUL, lw=1.8, marker="o", ms=3.5, mec=SURFACE, mew=0.8, zorder=3)
a.set_title("Focos de queimada (INPE)", loc="left", fontsize=9.5, fontweight="bold", color=INK)
b.set_title("Internações respiratórias (SIH, pela data de internação)", loc="left", fontsize=9.5, fontweight="bold", color=INK)
for an in anos:                                                  # pico anual: rótulo direto
    s = r[r.ano == an]
    pf, pi = s.loc[s.qtd_focos.idxmax()], s.loc[s.qtd_internacoes.idxmax()]
    a.annotate(f"{MESES[int(pf.mes) - 1].lower()}.: {milhar(pf.qtd_focos)}", (pf.periodo, pf.qtd_focos), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color=INK2)
    b.annotate(f"{MESES[int(pi.mes) - 1].lower()}.: {milhar(pi.qtd_internacoes)}", (pi.periodo, pi.qtd_internacoes), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color=INK2)
for ax in (a, b):
    ax.yaxis.set_major_formatter(milhar)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.10)
a.set_ylabel("focos por mês"); b.set_ylabel("internações por mês")
ticks = [(an - ano_min) * 12 + m for an in anos for m in (1, 4, 7, 10)]
b.set_xticks(ticks); b.set_xticklabels([f"{['jan', 'abr', 'jul', 'out'][i]}/{str(an)[-2:]}" for an in anos for i in range(4)], fontsize=7.5)
b.set_xlim(0.2, len(r) + 0.8)
fig.tight_layout(h_pad=1.4)
fig.savefig("fig/figura1_sazonalidade.png", dpi=220); plt.close(fig)

# ------------------------------------------------------------ Figura 2 (R² por divisão e configuração)
res = pd.read_csv("resultados_modelos_multiano.csv")
divs = list(res["divisão"].unique())
rotulos_div = ["Aleatória 80/20", "Por município", f"Por tempo\n(teste: {max(anos)})"]
def melhor(prefixos):
    """Melhor R² (entre Random Forest e GBT) por divisão, para os conjuntos de variáveis indicados."""
    out = []
    for dv in divs:
        s = res[(res["divisão"] == dv) & res["modelo"].isin(["Random Forest", "GBT"]) & res["variáveis"].str.startswith(prefixos)]
        out.append(float(s["r2"].max()))
    return out
def base(nome):
    return [float(res[(res["divisão"] == dv) & (res["modelo"] == nome)]["r2"].iloc[0]) for dv in divs]
series = [("Melhor modelo com focos (A ou B)", melhor(("A:", "B:")), LARANJA),
          ("Melhor modelo sem focos (C)", melhor(("C:",)), AZUL),
          ("Base: média do município", base("Base: média do município"), AMARELO),
          ("Base: município + mês (sazonal)", base("Base: município + mês (sazonal)"), CINZA)]
fig, ax = plt.subplots(figsize=(6.3, 3.9))
estilo(ax)
larg = 0.18; x0 = np.arange(len(divs)); n = len(series)
for i, (nome, vals, cor) in enumerate(series):
    xs = x0 + (i - (n - 1) / 2) * (larg + 0.015)
    ax.bar(xs, vals, width=larg, color=cor, label=nome, zorder=3)
    for xx, v in zip(xs, vals):
        t = f"{v:.2f}".replace(".", ",").replace("-0,00", "0,00")
        ax.text(xx, v + (0.012 if v >= 0 else -0.012), t, ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5, color=INK2)
ax.axhline(0, color=INK2, lw=0.9, zorder=4)
ax.set_xticks(x0); ax.set_xticklabels(rotulos_div)
ax.set_ylabel("R² no conjunto de teste")
ax.set_ylim(-0.18, 0.60)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=8, handlelength=1.2, columnspacing=1.2)
fig.tight_layout()
fig.savefig("fig/figura2_r2.png", dpi=220); plt.close(fig)

# ------------------------------------------------------------ Figura 3 (importância das variáveis, GBT, conjunto B, divisão por tempo)
# valores impressos no gráfico do notebook (importância em %), GBT, conjunto A, divisão por tempo
imp = {"População": 43.4, "Taxa de focos no mês (T)": 14.5, "Mês (sazonalidade)": 13.2,
       "Taxa de focos em T−1": 11.9, "Taxa de focos em T−2": 11.0, "Grupo 0 (dummy)": 3.8, "Grupo 2 (dummy)": 2.2, "Grupo 1 (dummy)": 0.0}
fig, ax = plt.subplots(figsize=(6.3, 3.6))
estilo(ax); ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRID, lw=0.8)
nomes = list(imp)[::-1]; vals = [imp[k] for k in nomes]
cores = [LARANJA if "focos" in k else AZUL for k in nomes]
ax.barh(nomes, vals, color=cores, height=0.62, zorder=3)
for i, v in enumerate(vals):
    ax.text(v + 0.7, i, f"{v:.1f}%".replace(".", ","), va="center", fontsize=8.5, color=INK2)
ax.set_xlabel("Importância (%)"); ax.set_xlim(0, 50)
ax.legend(handles=[Patch(color=LARANJA, label="Focos de calor"), Patch(color=AZUL, label="Demais variáveis")],
          loc="lower right", frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig("fig/figura3_importancia.png", dpi=220); plt.close(fig)

# ------------------------------------------------------------ Figura 4 (recorte do painel direito do gráfico 2 do notebook)
im = Image.open("grafico2_modelo_multiano.png")
w, h = im.size
im.crop((int(w * 0.525), 0, w, h)).save("fig/figura4_real_previsto.png")

# ------------------------------------------------------------ Figura A1 (padrão mensal médio por grupo)
p = pd.read_csv("base_painel_multiano.csv").dropna(subset=["populacao"])
cl = pd.read_csv("perfil_municipios_multiano.csv")[["cod_ibge_6", "cluster"]]
p = p.merge(cl, on="cod_ibge_6")
g = p.groupby(["cluster", "ano", "mes"]).agg(f=("qtd_focos", "sum"), i=("qtd_internacoes", "sum"), pop=("populacao", "sum")).reset_index()
g["tf"] = g.f / g["pop"] * 1e5; g["ti"] = g.i / g["pop"] * 1e5
g = g.groupby(["cluster", "mes"])[["tf", "ti"]].mean().reset_index()            # média dos anos, por mês do ano
nmun = cl.cluster.value_counts().to_dict()
fig, axs = plt.subplots(2, 3, figsize=(6.3, 4.6), sharex=True)
ymax_int = g["ti"].max() * 1.1
for j, c in enumerate([0, 1, 2]):
    s = g[g.cluster == c]
    for k, (col, cor, tit) in enumerate([("tf", LARANJA, "focos / 100 mil hab."), ("ti", AZUL, "internações / 100 mil hab.")]):
        ax = axs[k, j]; estilo(ax); ax.axvspan(6.5, 11.5, color="#efeeea", zorder=0)
        ax.plot(s.mes, s[col], color=cor, lw=2, marker="o", ms=4.5, mec=SURFACE, mew=1, zorder=3)
        if k == 0: ax.set_title(f"Grupo {c} ({nmun[c]} município{'s' if nmun[c] > 1 else ''})", fontsize=8.5, fontweight="bold", color=INK)
        if j == 0: ax.set_ylabel(tit, fontsize=8)
        ax.set_xticks([1, 3, 5, 7, 9, 11]); ax.set_xticklabels(["Jan", "Mar", "Mai", "Jul", "Set", "Nov"], fontsize=8)
        ax.set_ylim(0, ymax_int if k == 1 else s[col].max() * 1.12)
        ax.tick_params(axis="y", labelsize=7.5)
fig.tight_layout(h_pad=1.0, w_pad=0.8)
fig.savefig("fig/figuraA1_clusters_mensal.png", dpi=220); plt.close(fig)
print("figuras ok")
