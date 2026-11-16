# Turning StatsPro into income

StatsPro makes money in three ways. Both are **off until you configure them**, so
nothing changes for visitors until you are ready.

| Stream | What it is | Setup |
|---|---|---|
| **Pro licence** | Basic statistics, charts, preprocessing, AI insights and model training stay free. **PDF report, cleaned CSV download, reproduce script, batch prediction and model download** unlock with a licence key. | ~20 min, below |
| **Services** | "Work with me" card on the About tab (email quote request). Already live. | none |
| **Sponsorship** | Optional "Sponsor" button in the sidebar. | set `SPONSOR_URL` |

## 1. Create the product (Lemon Squeezy)

Lemon Squeezy is a merchant of record: it collects the money, handles sales tax
and pays you out. Its docs list **Bangladesh among bank-payout countries**
(PayPal payouts also cover 200+ countries). Check the current terms before you
commit, and note the minimum payout (about $50).

1. Create an account and a store at lemonsqueezy.com.
2. Add a product, e.g. **StatsPro Pro** — a one-time price (e.g. $19) or a
   yearly subscription. Turn on **Generate license keys**. Set **activation limit**
   high or unlimited (StatsPro validates keys, it does not count activations).
3. Copy the product's **checkout link**, your **store id** and **product id**
   (shown in the dashboard / product URL).

## 2. Configure the live app

Streamlit Cloud → your app → **Settings → Secrets**, paste:

```toml
CHECKOUT_URL  = "https://yourstore.lemonsqueezy.com/buy/xxxxxxxx"
LS_STORE_ID   = "12345"
LS_PRODUCT_ID = "67890"
# optional
SPONSOR_URL   = "https://github.com/sponsors/Junaid-Ahmed-Rupok"
HIRE_EMAIL    = "junaidahmedrupok@gmail.com"
```

Save. The sidebar now shows **🔑 Pro license** and the Pro features show an
upgrade card. Buyers get their key by email from Lemon Squeezy and paste it into
the sidebar. A key is accepted only if it was issued for *your* store and product.

Test it once yourself with a Lemon Squeezy test-mode purchase before announcing.

## Honest limits

- The code is open source, so a technical user can remove the check from their own
  copy. Pro sells convenience and the hosted app, not secrecy.
- Free hosting (Streamlit Community Cloud) is not meant for a busy paid product. If
  sales grow, move to a paid host.
- Keys are checked online (cached for an hour). If Lemon Squeezy is unreachable the
  user sees a "try again" message rather than a false "invalid".
- Keep the free tier genuinely useful — it is what brings people in and it is
  what reviewers see.

## Getting customers

Most income from a tool like this comes from being found, not from the paywall:
post the live app and a short demo on LinkedIn, in thesis / research communities
and on Reddit (r/statistics, r/datascience, r/AskStatistics, Bangladeshi
student groups); list a "data analysis" gig on Upwork/Fiverr that links to it; and
keep the "Work with me" card current.


## Student pricing experiment (Bangladesh)

Suggested prices to **test**, not validated market prices. Prices live in Lemon
Squeezy, not in the code, so changing them never needs a redeploy.

| Offer | Price to try | What it unlocks |
|---|---|---|
| Free | ৳0 | Upload, EDA, tests, charts, ML training |
| Single research export | ৳99–149 | PDF report, cleaned CSV and reproduce script for one project |
| 7-day student pass | ৳199 | All Pro features for a deadline week |
| Monthly student Pro | ৳299 | All Pro features |

Notes:
- Create each as a separate Lemon Squeezy product or variant. For passes, set the
  licence key to expire after 7 or 30 days. StatsPro only checks that a key is
  valid, so key expiry is enforced by Lemon Squeezy, not by this app.
- Confirm which currencies Lemon Squeezy lets you charge in. If BDT is not offered
  you will price in USD, so ৳ amounts above are only a guide.
- Track: visitors, uploads, completed analyses, clicks on "upgrade", paying users.
  The number that matters most is the share of uploaders who finish an analysis.
- Do not market StatsPro as guaranteeing correct conclusions or thesis acceptance.
  The honest selling point is transparent, reproducible analysis.


## Pay by bKash (manual, no fees to a platform)

Lemon Squeezy charges 5% + 50 cents per sale, which makes sub-$3 prices
pointless. For taka-sized prices (৳99–299) sell by **bKash Send Money** and issue
the key yourself. No payment API or merchant account is needed.

**One-time setup** (Streamlit Cloud, Settings, Secrets):

```toml
BKASH_NUMBER   = "01XXXXXXXXX"
LICENSE_SECRET = "make-up-a-long-random-string-at-least-16-chars"   # keep private
BKASH_PLANS    = "7 days: 199 BDT | 30 days: 299 BDT"               # shown to buyers
BKASH_CONTACT  = "junaidahmedrupok@gmail.com"                        # optional
WHATSAPP_NUMBER = "8801XXXXXXXXX"                                   # optional, adds a WhatsApp button (country code, no +)
```

`LICENSE_SECRET` never goes in the repo. Anyone who knows it can make keys, so
keep it only in Secrets and on your own computer.

**Each sale (easiest way on Windows):** double-click `newkey.bat`, type 7 or 30, and the key is shown and copied to your clipboard. The first time it asks for your `LICENSE_SECRET` and saves it in `.license_secret` on your computer only (git ignores that file).

**Each sale, manual commands:**
1. The buyer opens **Pay with bKash** in the app, sends the money, and emails you
   the TrxID and their email (a pre-filled email button is provided).
2. Check the payment arrived in your bKash app.
3. On your computer, from this repo:
   ```bash
   LICENSE_SECRET="your-secret" python tools/make_key.py --days 7
   ```
   (Windows PowerShell: `$env:LICENSE_SECRET="your-secret"; python tools/make_key.py --days 7`)
4. Email the key to the buyer. They paste it in the sidebar, 🔑 Pro license.

Keys are signed and carry an expiry date, so a 7-day pass really expires and the
app needs no server to check it. For a refund add the key to `REVOKED_KEYS`
(comma-separated) in Secrets.

Limits: it is manual, so every sale needs you to respond, and bKash gives no
automatic proof of payment inside the app. Always confirm the money in your own
bKash app before sending a key. Do not change `LICENSE_SECRET` after selling keys,
or every key issued so far stops working.


## Stopping a customer (refund, abuse)

1. Streamlit Cloud, your app, Settings, Secrets.
2. Add or extend the line, with the customer's key (comma-separated for several):
   ```toml
   REVOKED_KEYS = "SP1-20261123-38U6-QIOVYB5QK5WI,SP1-..."
   ```
3. Save. Within a minute the app restarts. Pro stops for that key, including for a
   person who is using the app right then, on their next click.

Keep a note of which key you gave to which buyer, otherwise you will not know which one
to cancel. (The `.license_secret` file never leaves your laptop; keep your own list of
keys and buyers in a private spreadsheet.)


## What is free and what is Pro

**Free:** upload, preprocessing, EDA, charts, the basic statistical tests (normality, correlation, chi-square), ML Studio training with
the model comparison and all the diagnostic charts (ROC, precision-recall, calibration, confusion
matrix, residuals, learning curve), the first 3 rows of the Assumption Doctor, a preview of the
write-up, and the Statistician Review verdict.

**Pro:** the advanced statistical tests (ANOVA, Mann-Whitney, Levene, multicollinearity/VIF, OLS regression), full Assumption Doctor results with explanations, the thesis write-up and reproduce
script, Statistician Review details and the sample-size planner, the ML Trust Report (prediction
intervals, fairness by group), live and batch prediction, model download, PDF report, cleaned CSV.

To move a feature between the tiers, change its `has_pro()` check in `app.py`.
