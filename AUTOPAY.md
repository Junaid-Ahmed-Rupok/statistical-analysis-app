# Automatic bKash key delivery

Goal: a buyer pays, types their TrxID into StatsPro, and gets a key instantly.

```
Buyer pays you on bKash
        |
bKash sends YOU an SMS ("You have received Tk 50.00 ... TrxID 9AB1C2D3EF")
        |  (SMS forwarder app on your Android phone, bKash messages only)
Google Sheet (free) stores TrxID + amount
        |
Buyer types TrxID in StatsPro  ->  StatsPro asks the Sheet, once  ->  key shown, Pro unlocked
```

You need: an Android phone that receives your bKash SMS, and a Google account.

## 1. Create the sheet and script (10 min)
1. Go to sheets.google.com and create a blank sheet named `StatsPro payments`.
2. **Extensions -> Apps Script**. Delete the sample code and paste the whole of
   `tools/apps_script/Code.gs` from this repo. Click Save.
3. **Project Settings -> Script properties -> Add**:
   - `INGEST_TOKEN` = a long random string (for your phone)
   - `CLAIM_TOKEN`  = a different long random string (for StatsPro)
4. **Deploy -> New deployment -> Web app**. Execute as **Me**, access **Anyone**.
   Authorise when asked. Copy the **Web app URL** (ends in `/exec`).

## 2. Connect StatsPro
Add to Streamlit Secrets (next to your other bKash settings):
```toml
AUTOPAY_URL    = "https://script.google.com/macros/s/.../exec"
AUTOPAY_TOKEN  = "<the CLAIM_TOKEN value>"
BKASH_PRICE_7  = "50"
BKASH_PRICE_30 = "99"
```

## 3. Forward the bKash SMS from your phone
Install an SMS forwarder that can POST to a URL and filter by sender (for example
the open-source "SMS Forwarder" apps). Configure:
- **Sender filter: bKash only.** This is essential. Without it, bank OTP codes could be
  forwarded. The script also refuses anything whose sender is not bKash.
- **URL:** the Web app URL from step 1.
- **Method:** POST with a JSON body, or GET with `?action=ingest&token=<INGEST_TOKEN>&from=<sender>&text=<message>` (for apps that only build links).
  (use the app's own placeholder names for sender and message).

## 4. Test it end to end
1. Ask a friend to send you the cheapest amount, or send yourself from a second number.
2. Check the sheet: a row with the TrxID and amount should appear within a minute.
3. In StatsPro, enter that TrxID under **Pay with bKash -> Get my key**.
4. You should see Pro unlock. Entering the same TrxID again must say it was already used.

If the row does not appear, the SMS wording may differ from what the script expects.
Open `parseSms_` in `Code.gs` and adjust the three patterns (amount, TrxID, sender number).

## Safety notes
- A TrxID works once, within 72 hours of the payment.
- Overpaying is fine; underpaying gets no key. 99+ BDT gets 30 days, 50+ gets 7 days.
- Nobody can forge a key without `LICENSE_SECRET`; nobody can claim without a real
  TrxID that reached your sheet.
- Confirm that using a personal bKash number for sales is allowed under bKash's terms,
  and consider a bKash Merchant account if volume grows.
- Never share `INGEST_TOKEN`, `CLAIM_TOKEN` or `LICENSE_SECRET`.
- If something goes wrong, buyers can still email you the TrxID and you can make a
  key by hand with `tools/make_key.py`.
