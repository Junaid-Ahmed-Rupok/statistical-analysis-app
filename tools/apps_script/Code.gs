/**
 * StatsPro bKash payment sheet (Google Apps Script).
 *
 * Phone (SMS forwarder)  --POST-->  doPost  : stores {TrxID, amount} in the sheet
 * StatsPro server        --GET--->  doGet   : claims a TrxID once and returns its amount
 *
 * Script properties (Project Settings -> Script properties):
 *   INGEST_TOKEN  secret the phone sends          (long random string)
 *   CLAIM_TOKEN   secret StatsPro sends (AUTOPAY_TOKEN in Streamlit secrets)
 * Use two different values. Never put them in GitHub.
 */

var SHEET_NAME = 'Payments';
var MAX_AGE_HOURS = 72;   // a payment can be claimed for 3 days

function sheet_() {
  var ss = SpreadsheetApp.getActive();
  var sh = ss.getSheetByName(SHEET_NAME);
  if (!sh) {
    sh = ss.insertSheet(SHEET_NAME);
    sh.appendRow(['TrxID', 'AmountBDT', 'From', 'ReceivedAt', 'ClaimedAt']);
  }
  return sh;
}

function prop_(name) {
  return PropertiesService.getScriptProperties().getProperty(name) || '';
}

function json_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}

/** Parse a bKash "received money" SMS. Returns null if it is not one. */
function parseSms_(text) {
  text = String(text || '');
  if (!/received/i.test(text)) return null;                    // ignore sent-money, cash-out, OTPs...
  var amount = text.match(/received\s+Tk\.?\s*([\d,]+(?:\.\d+)?)/i);
  var trx = text.match(/TrxID\s*[:\-]?\s*([A-Z0-9]{8,14})/i);
  var from = text.match(/from\s+(\+?\d{10,14})/i);
  if (!amount || !trx) return null;
  return {
    amount: parseFloat(amount[1].replace(/,/g, '')),
    trx: trx[1].toUpperCase(),
    from: from ? from[1] : ''
  };
}

/** Phone -> sheet. Accepts JSON {"from","text","token"} or form fields of the same names. */
function doPost(e) {
  var p = {};
  try { p = JSON.parse(e.postData.contents); } catch (err) { p = e.parameter || {}; }
  var token = p.token || (e.parameter && e.parameter.token) || '';
  if (!prop_('INGEST_TOKEN') || token !== prop_('INGEST_TOKEN')) return json_({ ok: false, error: 'bad_token' });

  // Only bKash. This stops bank OTPs or other SMS from ever being stored.
  if (!/bkash/i.test(String(p.from || ''))) return json_({ ok: false, error: 'not_bkash' });

  var sms = parseSms_(p.text);
  if (!sms) return json_({ ok: false, error: 'not_a_payment' });

  var lock = LockService.getScriptLock();
  lock.waitLock(20000);
  try {
    var sh = sheet_();
    var ids = sh.getRange(2, 1, Math.max(sh.getLastRow() - 1, 1), 1).getValues();
    for (var i = 0; i < ids.length; i++) if (String(ids[i][0]).toUpperCase() === sms.trx) return json_({ ok: true, dup: true });
    sh.appendRow([sms.trx, sms.amount, sms.from, new Date(), '']);
  } finally { lock.releaseLock(); }
  return json_({ ok: true });
}

/** StatsPro -> sheet. ?action=claim&trx=ID&token=CLAIM_TOKEN */
function doGet(e) {
  var q = e.parameter || {};
  if (q.action !== 'claim') return json_({ ok: false, error: 'bad_request' });
  if (!prop_('CLAIM_TOKEN') || q.token !== prop_('CLAIM_TOKEN')) return json_({ ok: false, error: 'bad_token' });
  var trx = String(q.trx || '').toUpperCase().replace(/[^A-Z0-9]/g, '');
  if (trx.length < 8) return json_({ ok: false, error: 'not_found' });

  var lock = LockService.getScriptLock();
  lock.waitLock(20000);
  try {
    var sh = sheet_();
    var n = sh.getLastRow() - 1;
    if (n < 1) return json_({ ok: false, error: 'not_found' });
    var rows = sh.getRange(2, 1, n, 5).getValues();
    for (var i = 0; i < rows.length; i++) {
      if (String(rows[i][0]).toUpperCase() !== trx) continue;
      if (rows[i][4]) return json_({ ok: false, error: 'already_used' });
      var ageH = (new Date() - new Date(rows[i][3])) / 3600000;
      if (ageH > MAX_AGE_HOURS) return json_({ ok: false, error: 'expired' });
      sh.getRange(i + 2, 5).setValue(new Date());           // mark used BEFORE replying
      return json_({ ok: true, amount: Number(rows[i][1]) });
    }
  } finally { lock.releaseLock(); }
  return json_({ ok: false, error: 'not_found' });
}
