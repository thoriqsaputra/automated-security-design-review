import httpx
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)

def scrape_url(url: str) -> str:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        with httpx.Client(headers=headers, follow_redirects=True, timeout=30.0, verify=False) as client:
            response = client.get(url)
            response.raise_for_status()
            
            redirect_count = 0
            while redirect_count < 5:
                soup = BeautifulSoup(response.text, "html.parser")
                meta_refresh = soup.find("meta", attrs={"http-equiv": "refresh"})
                if not meta_refresh:
                    break
                
                import re
                try:
                    content = meta_refresh["content"]
                    match = re.search(r'url=[\'"]?([^\'"]+)', content, re.I)
                    if match:
                        next_url = httpx.URL(url).join(match.group(1))
                        url = next_url # Update current URL
                        response = client.get(url)
                        response.raise_for_status()
                        redirect_count += 1
                    else:
                        break
                except Exception as meta_e:
                    logger.warning(f"Failed to follow meta refresh url for {url}: {meta_e}")
                    break

    except httpx.HTTPError as e:
        logger.error(f"Error scraping URL {url}: {e}")
        return ""
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    for script in soup(["script", "style", "header", "footer", "nav"]):
        script.extract()
        
    text = soup.get_text(separator="\n")
    lines = (line.strip() for line in text.splitlines())
    text_content = "\n".join(line for line in lines if line)
    
    return text_content
