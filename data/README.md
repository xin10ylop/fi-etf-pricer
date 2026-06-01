# Local holdings fallback

Drop a manually downloaded holdings CSV here when a provider CDN blocks the live
fetch from your host (a 403 Forbidden or an HTML bot-challenge page). The loader
uses a local file in preference to a live download, for any provider.

Name the file by ticker, either form works:

    data/GOVT.csv
    data/GOVT_holdings.csv

The file must be the provider's raw holdings CSV (the same content the live
endpoint serves), not a saved HTML page. The loader detects HTML and refuses it.

This works identically for iShares, Vanguard, and State Street (SSGA): all three
share the same browser-header download path, and all three accept a local file
fallback here.
