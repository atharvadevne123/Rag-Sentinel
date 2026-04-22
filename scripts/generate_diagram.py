import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

os.makedirs("screenshots", exist_ok=True)


def draw_box(ax, x, y, w, h, label, color, fontsize=9, text_color="white"):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                         linewidth=1.5, edgecolor="white", facecolor=color, zorder=3)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color, zorder=4,
            wrap=True, multialignment="center")


def draw_arrow(ax, x1, y1, x2, y2, color="#cccccc"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5), zorder=2)


fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_facecolor("#1a1a2e")
fig.patch.set_facecolor("#1a1a2e")
ax.axis("off")

fig.suptitle("RAG Sentinel — System Architecture", fontsize=16, fontweight="bold",
             color="white", y=0.97)

# Client layer
draw_box(ax, 0.3, 7.2, 2.2, 0.8, "Client /\nHTTP Request", "#4a4e69", fontsize=8)

# API Gateway
draw_box(ax, 3.0, 7.2, 2.5, 0.8, "FastAPI Gateway\n/predict /ingest /metrics", "#22577a", fontsize=8)

# Anomaly detection
draw_box(ax, 6.2, 7.2, 2.8, 0.8, "Anomaly Detection\nRF + IsolationForest", "#38a3a5", fontsize=8)

# Feature engineering
draw_box(ax, 6.2, 5.8, 2.8, 0.9, "Feature Engineering\n15 features: lag, rolling,\nratios, lexical", "#57cc99", fontsize=7.5, text_color="#1a1a2e")

# RAG pipeline
draw_box(ax, 3.0, 5.3, 2.5, 1.0, "RAG Pipeline\nRetriever → Context\n→ Extractive Answer", "#c77dff", fontsize=7.5)

# FAISS index
draw_box(ax, 0.3, 5.3, 2.2, 1.0, "FAISS Index\n+ Chunk Store\n(Hashed TF-IDF)", "#7b2d8b", fontsize=7.5)

# Ingest
draw_box(ax, 0.3, 3.5, 2.2, 0.9, "Document Ingest\nClean → Chunk →\nEmbed → Index", "#9d4edd", fontsize=7.5)

# Monitoring
draw_box(ax, 10.0, 7.2, 2.5, 0.8, "Model Monitoring\nKS-drift + Logs", "#e07a5f", fontsize=8)

# Database
draw_box(ax, 10.0, 5.5, 2.5, 1.0, "Database\nSQLAlchemy\nSQLite / PostgreSQL", "#3d405b", fontsize=7.5)

# Airflow DAG
draw_box(ax, 10.0, 3.8, 2.5, 1.0, "Airflow DAG\nDaily: drift check\n→ auto-retrain", "#f2cc8f", fontsize=7.5, text_color="#1a1a2e")

# ML Training
draw_box(ax, 6.2, 3.8, 2.8, 1.0, "ML Training\n5-fold CV + AUC-ROC\nmodel.joblib", "#80b918", fontsize=7.5, text_color="#1a1a2e")

# Docker layer box
docker_box = FancyBboxPatch((0.1, 3.0), 14.8, 5.5, boxstyle="round,pad=0.1",
                             linewidth=2, edgecolor="#00b4d8", facecolor="none",
                             linestyle="--", zorder=1)
ax.add_patch(docker_box)
ax.text(7.5, 3.05, "Docker Compose Environment", ha="center", va="bottom",
        fontsize=8, color="#00b4d8", style="italic")

# GitHub Actions box
ci_box = FancyBboxPatch((0.1, 1.5), 7.0, 1.3, boxstyle="round,pad=0.1",
                          linewidth=1.5, edgecolor="#f4a261", facecolor="#2d1b00",
                          linestyle="--", zorder=1)
ax.add_patch(ci_box)
draw_box(ax, 0.3, 1.6, 3.0, 1.0, "GitHub Actions CI\nruff lint → pytest", "#f4a261", fontsize=8, text_color="#1a1a2e")
draw_box(ax, 3.8, 1.6, 2.8, 1.0, "Pytest Suite\n5 test modules\nmocked externals", "#e9c46a", fontsize=8, text_color="#1a1a2e")

# Arrows
draw_arrow(ax, 2.5, 7.6, 3.0, 7.6)
draw_arrow(ax, 5.5, 7.6, 6.2, 7.6)
draw_arrow(ax, 7.6, 7.2, 7.6, 6.7)
draw_arrow(ax, 6.2, 6.25, 5.5, 6.0)
draw_arrow(ax, 3.0, 5.8, 2.5, 5.8)
draw_arrow(ax, 1.4, 5.3, 1.4, 4.4)
draw_arrow(ax, 9.0, 7.6, 10.0, 7.6)
draw_arrow(ax, 11.25, 7.2, 11.25, 6.5)
draw_arrow(ax, 11.25, 5.5, 11.25, 4.8)
draw_arrow(ax, 10.0, 4.3, 9.0, 4.3)
draw_arrow(ax, 7.6, 5.8, 7.6, 4.8)

legend_patches = [
    mpatches.Patch(color="#22577a", label="API Layer"),
    mpatches.Patch(color="#38a3a5", label="ML / Anomaly"),
    mpatches.Patch(color="#c77dff", label="RAG Layer"),
    mpatches.Patch(color="#e07a5f", label="Monitoring"),
    mpatches.Patch(color="#3d405b", label="Persistence"),
    mpatches.Patch(color="#f2cc8f", label="Orchestration"),
]
ax.legend(handles=legend_patches, loc="lower right", fontsize=8,
          facecolor="#1a1a2e", edgecolor="white", labelcolor="white",
          bbox_to_anchor=(0.99, 0.02))

plt.tight_layout()
plt.savefig("screenshots/architecture.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Architecture diagram saved to screenshots/architecture.png")
