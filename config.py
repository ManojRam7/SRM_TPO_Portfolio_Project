"""
Central configuration for the revenue management and trade promotion optimisation pipeline.

The unit of analysis is CTA x PPG x week (Census Trading Area x Product Price Group x
retail week), the standard grain for scanner-data demand modelling in consumer goods.
All brands, manufacturers and parameters below are fictional.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
CHARTS = OUT / "charts"
for _p in (DATA, OUT, CHARTS):
    _p.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 1234          # fixed so every run reproduces the same data and results
N_WEEKS = 156               # three years of weekly data
START_DATE = "2021-01-04"   # first Monday

# Census Trading Areas (markets). A national model would use many more; five keeps runs fast.
CTAS = ["GreatLakes", "Northeast", "South", "West", "Plains"]

# The manufacturer whose promotions are being optimised. Everyone else is competition.
FOCAL_MANUFACTURER = "CRESTLINE"

# ---------------------------------------------------------------------------
# Product Price Groups (PPGs). Each has known ground-truth parameters so the
# modelling stage can be validated: a correct log-log model must recover these.
# beta_price = true own-price elasticity (negative).
# ---------------------------------------------------------------------------
PPGS = [
    # ppg_id                    manufacturer   brand             category     segment     base_price  beta_price  beta_feat  beta_disp  base_units
    ("NUTTY_CARAMEL_SINGLE",    "CRESTLINE",   "Nutty Caramel",  "Chocolate", "Singles",  1.29,  -2.6,  0.45,  0.30, 9000),
    ("TOFFEE_BISCUIT_SINGLE",   "CRESTLINE",   "Toffee Biscuit", "Chocolate", "Singles",  1.29,  -2.3,  0.40,  0.28, 7000),
    ("HAPPY_BUTTONS_PEG",       "CRESTLINE",   "Happy Buttons",  "Chocolate", "Sharebag", 3.49,  -1.9,  0.55,  0.42, 6000),
    ("BLISS_SILK_BAR",          "CRESTLINE",   "Bliss Silk",     "Chocolate", "Singles",  1.49,  -2.8,  0.38,  0.25, 4000),
    ("GOLDEN_MALT_POUCH",       "CRESTLINE",   "Golden Malt",    "Chocolate", "Sharebag", 3.29,  -2.1,  0.50,  0.40, 3500),
    ("ARCTIC_MINT_GUM",         "CRESTLINE",   "Arctic Mint",    "Gum",       "Singles",  1.79,  -1.6,  0.30,  0.18, 5000),
    ("ICE_BREEZE_GUM",          "CRESTLINE",   "Ice Breeze",     "Gum",       "Singles",  1.79,  -1.7,  0.28,  0.16, 3000),
    ("JUICY_CHEWS_PEG",         "CRESTLINE",   "Juicy Chews",    "Fruity",    "Sharebag", 3.19,  -2.0,  0.48,  0.36, 4500),
    # Competitors (other manufacturers)
    ("CLASSIC_MILK_SINGLE",     "BROOKFIELD",  "Classic Milk",   "Chocolate", "Singles",  1.25,  -2.5,  0.42,  0.30, 8000),
    ("CRISPY_WAFER_SINGLE",     "ALPINE",      "Crispy Wafer",   "Chocolate", "Singles",  1.29,  -2.4,  0.40,  0.29, 7500),
    ("DELUXE_TRUFFLE_SHAREBAG", "SWISSGOLD",   "Deluxe Truffle", "Chocolate", "Sharebag", 4.49,  -1.8,  0.35,  0.25, 2500),
    ("SMOOTH_MINT_GUM",         "HARTWELL",    "Smooth Mint",    "Gum",       "Singles",  1.69,  -1.7,  0.27,  0.15, 4000),
    ("RAINBOW_SQUARES_PEG",     "KESTREL",     "Rainbow Squares","Fruity",    "Sharebag", 3.09,  -2.1,  0.46,  0.34, 4200),
    ("CHEWY_BEARS_PEG",         "BAYLEAF",     "Chewy Bears",    "Fruity",    "Sharebag", 2.99,  -2.2,  0.44,  0.33, 3800),
]
PPG_COLUMNS = ["ppg", "manufacturer", "brand", "category", "segment",
               "base_price", "beta_price", "beta_feat", "beta_disp", "base_units"]

# Cross-price (cannibalisation/competition) ground truth: (target, actor, beta)
# beta_cross > 0  =>  when the ACTOR's price goes up, the TARGET sells more.
CROSS_PRICE_TRUTH = [
    ("NUTTY_CARAMEL_SINGLE",  "TOFFEE_BISCUIT_SINGLE",   0.35),   # intra: own-portfolio cannibalisation
    ("NUTTY_CARAMEL_SINGLE",  "CLASSIC_MILK_SINGLE",     0.55),   # inter: competitor
    ("TOFFEE_BISCUIT_SINGLE", "NUTTY_CARAMEL_SINGLE",    0.33),
    ("TOFFEE_BISCUIT_SINGLE", "CRISPY_WAFER_SINGLE",     0.50),
    ("BLISS_SILK_BAR",        "CLASSIC_MILK_SINGLE",     0.45),
    ("HAPPY_BUTTONS_PEG",     "JUICY_CHEWS_PEG",         0.30),   # intra
    ("HAPPY_BUTTONS_PEG",     "DELUXE_TRUFFLE_SHAREBAG", 0.40),   # inter
    ("ARCTIC_MINT_GUM",       "ICE_BREEZE_GUM",          0.38),   # intra
    ("ARCTIC_MINT_GUM",       "SMOOTH_MINT_GUM",         0.42),   # inter
    ("JUICY_CHEWS_PEG",       "CHEWY_BEARS_PEG",         0.40),
]

# Pantry-loading: a discount this week depresses demand in the following week.
PANTRY_BETA = -0.012     # applied to lagged discount depth (%)
TREND_BETA = 0.0008      # mild positive trend per week
HOLIDAY_UPLIFT = {"Halloween": 0.55, "Christmas": 0.45, "Easter": 0.35,
                  "Valentine": 0.25, "SummerDip": -0.20}
NOISE_SD = 0.06          # lognormal noise on units

# ---- Financials (for ROI and the optimiser) ----
LIST_PRICE_UPLIFT = 1.55   # list price = base shelf price x this (wholesale-to-retail proxy)
GMARGIN = 0.42             # gross margin fraction on net sales
TRADE_RATE = 0.18          # trade spend as a share of gross sales value (budget guardrail)

# ---- Modelling controls ----
TEST_FRACTION = 0.20       # hold out the last 20% of weeks per CTA-PPG (time-based)
VIF_THRESHOLD = 8.0        # drop competitor terms above this variance inflation factor
AIC_IMPROVE_MIN = 2.0      # minimum AIC improvement to drop a term in stepwise selection
# Economic sign priors used to reject counter-intuitive coefficients
SIGN_PRIORS = {
    "log_price": -1,       # own price must be negative
    "feat_acv": +1,
    "disp_acv": +1,
    "si_log": +1,
    "trend": +1,
    "pantry_lag": -1,
}
# Plausible coefficient bounds (rejected if outside)
PRICE_BETA_BOUNDS = (-10.0, 0.0)
CROSS_BETA_BOUNDS = (0.0, 1.5)

# ---- Optimiser guardrails ----
MIN_NO_PROMO_WEEKS = 30        # minimum number of weeks at base price
MAX_PROMO_DEPTH = 0.40         # 40% maximum discount
MIN_AUP_FRACTION = 0.70        # average unit price floor as a share of base price
OFFER_GRID = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]  # candidate discount depths
