# IEEE Research Paper: Bharat AI Innovation Challenge

This directory contains the production IEEE conference LaTeX research paper documenting the Edge-AI Industrial Quality Inspection Cyber-Physical System.

## File List

- **`bharat_ai_ieee_research_paper.tex`**: Complete, self-contained 4-page IEEE conference format LaTeX document (`IEEEtran`).
- **Markdown Preview Artifact**: [bharat_ai_ieee_research_paper.md](file:///C:/Users/AMAL/.gemini/antigravity/brain/3ebc73c5-2d80-4f34-ac5c-af3c89343a36/bharat_ai_ieee_research_paper.md) in the Antigravity artifact directory.

## How to Compile to PDF

### Option A: Overleaf (Recommended - One Click)
1. Open [Overleaf.com](https://www.overleaf.com).
2. Create a **New Project** $\rightarrow$ **Blank Project**.
3. Replace the contents of `main.tex` with the contents of [`bharat_ai_ieee_research_paper.tex`](file:///c:/Users/AMAL/Defect-classifications/publication/bharat_ai_ieee_research_paper.tex).
4. Click **Recompile**. (Overleaf provides `IEEEtran.cls` by default).

### Option B: Local TeX Live / MiKTeX (Command Line)
Run the following in PowerShell / terminal:
```bash
cd publication
pdflatex bharat_ai_ieee_research_paper.tex
pdflatex bharat_ai_ieee_research_paper.tex
```

## Adding Your Physical Setup and Collage Images

In `bharat_ai_ieee_research_paper.tex`:
- Replace `\begin{tikzpicture} ... [ PLACEHOLDER: EXPERIMENTAL SETUP & RIG DIAGRAM ] ... \end{tikzpicture}` with:
  ```latex
  \includegraphics[width=\columnwidth]{your_camera_rig_photo.jpg}
  ```
- Replace `\begin{tikzpicture} ... [ PLACEHOLDER: QUALITATIVE DETECTION COLLAGE ] ... \end{tikzpicture}` with:
  ```latex
  \includegraphics[width=\textwidth]{your_qualitative_collage.jpg}
  ```
