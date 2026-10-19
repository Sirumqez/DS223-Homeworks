
# Marketing Analytics: Innovation Diffusion Analysis (Bass Model)

**Author:** Sereen Sipan  
**Course:** Marketing Analytics  
**Instructor:** Karen Hovhannisyan 
** Teaching Assistant:** Artur Avagyan
---

## Executive Summary
This project analyzes the diffusion path of the **Lotus Ring** (TIME's Best Inventions of 2025) using the **Bass Diffusion Model**. Parameters ($p$, $q$, $M$) were estimated via non-linear least squares using global smart wearable shipment data from Statista (CCS Insight).

---

## Project Structure
```
Homework1/
├── data/
│   └── smart_wearable_shipments_forecast_worldwide_from_2016_to_2026_in_millions.csv
├── img/
│   └── diffusion_plot.png
├── report/
│   ├── report.md
│   └── report.pdf
├── bass_analysis.py
└── README.md

```

---

## Key Results

* **Selected Innovation:** Lotus Ring ([TIME Best Inventions 2025](https://time.com/collections/best-inventions-2025/7318226/lotus-ring/))
* **Look-Alike Historical Category:** Global Smart Wearables / Smart Wristwear (CCS Insight / Statista)
* **Estimated Bass Parameters:**
* **Coefficient of Innovation ($p$):** `0.01406`
* **Coefficient of Imitation ($q$):** `0.21152`
* **Market Potential ($M$):** `7,272.17 million units`



---

## How to Run

1. **Set up Virtual Environment & Install Dependencies:**
```bash
pip install numpy pandas scipy matplotlib

```


2. **Run the Bass Analysis Script:**
```bash
python bass_analysis.py

```

*This script fits the Bass Model, prints parameter estimations and diffusion predictions, and updates `img/diffusion_plot.png`.*
