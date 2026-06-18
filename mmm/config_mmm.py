"""
Marketing mix modelling (MMM) configuration: the retailer-level workstream.
Unit of analysis: Retailer x Promo Group (PG) x Week, for two fictional grocery chains.

As in the SRM module, every promo group has known true coefficients so the
elastic-net model's recovery can be validated. All names are fictional.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
import config as C  # reuse N_WEEKS, START_DATE, RANDOM_SEED, GMARGIN

MMM_DATA = ROOT / "mmm" / "data"
MMM_OUT = ROOT / "mmm" / "outputs"
MMM_CHARTS = MMM_OUT / "charts"
for _p in (MMM_DATA, MMM_OUT, MMM_CHARTS):
    _p.mkdir(parents=True, exist_ok=True)

RETAILERS = ["Freshway", "Northmart"]

# promo_group           manufacturer  brand        segment        base_price  e_price  b_catalogue  b_display  base_qty
PROMO_GROUPS = [
    ("HONEY_NOUGAT_SINGLE",  "CRESTLINE", "Honey Nougat",   "Single",         2.00, -2.4, 0.50, 0.30, 7000),
    ("BITES_BUTTONS_POUCH",  "CRESTLINE", "Happy Buttons",  "Bitesize Pouch", 5.00, -2.0, 0.60, 0.40, 5000),
    ("BITES_MALT_POUCH",     "CRESTLINE", "Golden Malt",    "Bitesize Pouch", 5.50, -2.2, 0.55, 0.38, 4500),
    ("GOLDEN_MALT_BLOCK",    "CRESTLINE", "Golden Malt",    "Block",          4.50, -2.1, 0.48, 0.34, 3000),
    ("CLASSIC_DAIRY_BLOCK",  "HARTWELL",  "Hartwell Dairy", "Block",          4.00, -2.5, 0.52, 0.33, 9000),
    ("SILKEN_BLOCK",         "ALPINE",    "Alpine Silken",  "Block",          4.20, -2.3, 0.50, 0.32, 6000),
    ("DARK_ORCHARD_BLOCK",   "ORCHARD",   "Orchard Dark",   "Block",          5.00, -1.9, 0.40, 0.28, 2500),
]
PG_COLS = ["promo_group", "manufacturer", "brand", "segment", "base_price",
           "e_price", "b_catalogue", "b_display", "base_qty"]

# Cross-price (acting item) ground truth: (target, actor, beta>0)
CROSS_TRUTH = [
    ("HONEY_NOUGAT_SINGLE",  "CLASSIC_DAIRY_BLOCK", 0.45),
    ("BITES_BUTTONS_POUCH",  "BITES_MALT_POUCH",    0.30),   # intra: own-portfolio
    ("BITES_BUTTONS_POUCH",  "CLASSIC_DAIRY_BLOCK", 0.40),   # inter: competitor
    ("BITES_MALT_POUCH",     "BITES_BUTTONS_POUCH", 0.28),   # intra
    ("GOLDEN_MALT_BLOCK",    "CLASSIC_DAIRY_BLOCK", 0.42),
    ("GOLDEN_MALT_BLOCK",    "SILKEN_BLOCK",        0.35),
]

PANTRY_BETA = -0.010
TREND_BETA = 0.0006
NOISE_SD = 0.07
CATALOGUE_RATE = 0.22     # share of weeks with a catalogue (feature) event
DISPLAY_RATE = 0.18

FOCAL_MANUFACTURER = C.FOCAL_MANUFACTURER

# ---- Trade terms for ROI (illustrative rates) ----
# TE (trade expenditure) per unit = list x std_terms + settlement + deferred deal
TRADE_TERMS = {
    "std_terms": 0.10, "settlement_discount": 0.02, "deferred_deal": 0.0,
}
LIST_UPLIFT = 0.85        # list (wholesale) price = shelf base * this
GMARGIN = C.GMARGIN

# ---- Elastic net ----
ENET_ALPHA = 0.8          # 1 = lasso, 0 = ridge; 0.8 leans towards sparse models
ENET_LAMBDAS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
ACTING_ITEM_CORR_CUTOFF = 0.05   # min |corr| to keep a cross-price candidate
RELAXED_REFIT = True      # refit OLS on selected features (removes shrinkage bias)
