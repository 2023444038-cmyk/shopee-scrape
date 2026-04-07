import requests
import re
import time
import random
import os

USER_AGENTS = [
    'com.shopee.my/2.94.21 (Android 12; Pixel 6)',
    'com.shopee.my/2.90.10 (Android 11; Samsung Galaxy S21)',
    'Mozilla/5.0 (Linux; Android 12; Pixel 6) AppleWebKit/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
]


def scrape_shopee_bulk(url, total_wanted=100):
    """
    Scrape Shopee reviews via API.
    Returns list of dicts: [{'review': str, 'star': int, 'variation': str}]
    """
    match = re.search(r'i\.(\d+)\.(\d+)', url)
    if not match:
        print("❌ Link tidak sah — tiada pattern i.SHOPID.ITEMID")
        return []

    shop_id, item_id = match.group(1), match.group(2)
    print(f"📦 Shop ID: {shop_id} | Item ID: {item_id} | Sasaran: {total_wanted} ulasan")

    extracted_data = []
    offset = 0
    limit = 50
    consecutive_failures = 0

    while len(extracted_data) < total_wanted:
        headers = {
            'User-Agent': random.choice(USER_AGENTS),
            'Referer': f'https://shopee.com.my/product-i.{shop_id}.{item_id}',
            'Host': 'shopee.com.my',
            'Accept': 'application/json',
            'X-API-Source': 'pc',
        }

        api_url = (
            f"https://shopee.com.my/api/v2/item/get_ratings"
            f"?filter=0&flag=1&itemid={item_id}"
            f"&limit={limit}&offset={offset}"
            f"&shopid={shop_id}&type=0"
        )

        try:
            # Random delay — elak pattern detection
            delay = random.uniform(2.0, 4.5)
            print(f"⏳ Tunggu {delay:.1f}s sebelum request...")
            time.sleep(delay)

            response = requests.get(api_url, headers=headers, timeout=20)

            if response.status_code == 200:
                data = response.json()

                # Shopee error code check
                if data.get('error') not in (0, None):
                    print(f"❌ Shopee API error code: {data.get('error')}")
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        print("Terlalu banyak kegagalan. Berhenti.")
                        break
                    continue

                ratings = data.get('data', {}).get('ratings', [])
                if not ratings:
                    print(f"🏁 Tiada lagi ulasan. Berhenti pada {len(extracted_data)}.")
                    break

                for r in ratings:
                    comment = r.get('comment', '').strip()
                    if not comment or len(comment) < 3:
                        continue

                    items = r.get('product_items', [])
                    variation_names = [i.get('model_name', '') for i in items if i.get('model_name')]
                    variation_str = ", ".join(variation_names) if variation_names else "No Variation"

                    star = r.get('rating_star', 5)
                    extracted_data.append({
                        'review': comment.replace('\n', ' '),
                        'star': int(star),
                        'variation': variation_str
                    })

                    if len(extracted_data) >= total_wanted:
                        break

                consecutive_failures = 0
                print(f"✅ {len(extracted_data)} / {total_wanted} ulasan diperolehi...")
                offset += limit

            elif response.status_code == 403:
                print(f"❌ 403 Forbidden — IP diblock oleh Shopee.")
                break

            else:
                print(f"❌ HTTP {response.status_code}")
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    break

        except requests.exceptions.Timeout:
            print("❌ Timeout — server lambat response")
            consecutive_failures += 1
            if consecutive_failures >= 3:
                break

        except Exception as e:
            print(f"❌ Error: {e}")
            consecutive_failures += 1
            if consecutive_failures >= 3:
                break

    print(f"✅ Selesai. Total: {len(extracted_data)} ulasan.")
    return extracted_data[:total_wanted]


if __name__ == "__main__":
    import pandas as pd

    url = "https://shopee.com.my/NEW%E3%80%90-199-Speed-%E3%80%91-Rechargeable-Mini-Small-Fan-i.1276527157.29484775921"
    results = scrape_shopee_bulk(url, total_wanted=100)

    if results:
        df = pd.DataFrame(results)
        out = "stage1_bulk_reviews.csv"
        if os.path.isfile(out):
            import pandas as pd
            existing = pd.read_csv(out)
            combined = pd.concat([existing, df], ignore_index=True)
            combined.drop_duplicates(subset=['review'], inplace=True)
            combined.to_csv(out, index=False, encoding='utf-8-sig')
            print(f"✅ Ditambah. Total: {len(combined)} ulasan.")
        else:
            df.to_csv(out, index=False, encoding='utf-8-sig')
            print(f"✅ Disimpan: {len(df)} ulasan.")
