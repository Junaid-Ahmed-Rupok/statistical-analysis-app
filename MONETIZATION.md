# Turning StatsPro into income

StatsPro makes money in two ways. Both are **off until you configure them**, so
nothing changes for visitors until you are ready.

| Stream | What it is | Setup |
|---|---|---|
| **Pro licence** | Statistics, charts, preprocessing, AI insights and model training stay free. **PDF report, cleaned CSV download, reproduce script, batch prediction and model download** unlock with a licence key. | ~20 min, below |
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
