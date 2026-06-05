import requests
from bs4 import BeautifulSoup
import re
import csv
import time

def parse_scorecard(match_code):
    """
    Fetches the scorecard page for a specific match code and extracts match details.
    """
    url = f"https://howstat.com/Cricket/statistics/matches/MatchScorecard_ODI.asp?MatchCode={match_code}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    # Retry logic
    for _ in range(3):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        return None
        
    soup = BeautifulSoup(response.text, 'html.parser')
    title_text = soup.title.get_text(strip=True) if soup.title else ""
    
    match_date = ""
    venue = ""
    team1 = "Unknown"
    team2 = "Unknown"
    match_result = "N/A"
    
    # 1. Date extraction
    date_match = re.search(r'\d{2}/\d{2}/\d{4}', title_text)
    if date_match:
        match_date = date_match.group(0)
        
    # 2. Team extraction from Title
    parts = title_text.split(" - ")
    for part in parts:
        if " v " in part or " v. " in part:
            split_char = " v " if " v " in part else " v. "
            t_parts = part.split(split_char)
            if len(t_parts) == 2:
                t1 = t_parts[0].strip()
                t2 = t_parts[1].strip()
                # Clean year prefixes
                t1 = re.sub(r'^\d{4}(-\d{4})?\s+', '', t1).strip()
                t2 = re.sub(r'^\d{4}(-\d{4})?\s+', '', t2).strip()
                team1 = t1
                team2 = t2
                break
                
    # 3. Venue extraction from table or fallback
    tables = soup.find_all('table')
    for table in tables:
        text = table.get_text()
        if "MATCH INFORMATION" in text:
            rows = table.find_all('tr')
            for row in rows:
                cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
                if len(cells) >= 2:
                    if cells[0] == "Venue":
                        venue = cells[1]
                        break
                        
    if not venue:
        if len(parts) >= 5:
            venue = parts[4].strip()
            
    # 4. Batsmen scores parsing
    batsmen_scores = []
    for table in tables:
        rows = table.find_all('tr')
        has_batting = False
        for row in rows:
            cells = [c.get_text(strip=True) for c in row.find_all(['td', 'th'])]
            if "BATTING" in cells and "R" in cells:
                has_batting = True
                break
        
        if has_batting:
            for row in rows:
                cells = [c.get_text(strip=True) for c in row.find_all(['td', 'th'])]
                if len(cells) >= 3:
                    player_name = cells[0]
                    if player_name in ["BATTING", "Extras", "TOTAL", "Did Not Bat", "Fall of Wickets", "BOWLING", "MATCH INFORMATION", "Venue", "Toss", "Series", "Match No.", "Match Conditions", "Player of Match", "Partnerships", "Scorecards Menu"]:
                        continue
                    if not player_name or len(player_name) < 3:
                        continue
                    runs_str = cells[2]
                    if runs_str.isdigit():
                        runs = int(runs_str)
                        clean_name = re.sub(r'[\*\u2020\d]', '', player_name).strip()
                        batsmen_scores.append((clean_name, runs))
                        
    # 5. Clean Result parsing
    text_content = soup.get_text()
    lines = text_content.split('\n')
    for line in lines:
        line_clean = line.strip()
        if any(k in line_clean for k in ["won by", "tied", "abandoned", "Match drawn", "Match Tied"]):
            if len(line_clean) > 10 and not any(x in line_clean for x in ["BATTING", "BOWLING", "Extras", "TOTAL"]):
                match_result = re.sub(r'\s+', ' ', line_clean).strip()
                break
                
    if batsmen_scores:
        batsmen_scores.sort(key=lambda x: x[1], reverse=True)
        top_scorer, top_score = batsmen_scores[0]
    else:
        top_scorer, top_score = "N/A", "N/A"
        
    return {
        "date": match_date,
        "team1": team1,
        "team2": team2,
        "venue": venue,
        "result": match_result,
        "top_scorer": top_scorer,
        "score": top_score
    }

def get_last_10_completed_matches(country_code, country_name):
    """
    Submits a POST request to list country ODI matches on HowStat,
    filters the last 10 completed matches, and parses their scorecards.
    """
    url = "https://howstat.com/Cricket/statistics/matches/MatchListCountry_ODI.asp"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    payload = {
        "cboCountry1": country_code,
        "cboPlayed": "XXX",
        "cboCountry2": "XXX",
        "cboFrom": "1971",
        "cboTo": "2026"
    }
    
    print(f"\nFetching ODI match list for {country_name} ({country_code})...")
    response = requests.post(url, headers=headers, data=payload)
    if response.status_code != 200:
        print(f"Failed to fetch matches for {country_code}")
        return []
        
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', class_='TableLined')
    if not table:
        print(f"Data table not found for {country_code}")
        return []
        
    rows = table.find_all('tr')
    completed_matches = []
    
    # Reverse iterate to find the most recent completed matches
    for row in reversed(rows[1:]):
        cells = row.find_all(['td', 'th'])
        if len(cells) < 5:
            continue
            
        result_text = cells[4].get_text(strip=True).lower()
        if any(kw in result_text for kw in ["abandoned", "no result", "cancelled"]):
            continue
            
        scorecard_cell = cells[2]
        link = scorecard_cell.find('a')
        if not link:
            continue
            
        href = link.get('href')
        match_code_m = re.search(r'MatchCode=(\d+)', href)
        if not match_code_m:
            continue
            
        match_code = match_code_m.group(1)
        completed_matches.append((match_code, country_name))
        if len(completed_matches) == 10:
            break
            
    print(f"Found 10 completed matches for {country_name}.")
    
    results = []
    for idx, (code, team_name) in enumerate(completed_matches, 1):
        print(f"[{team_name}] Parsing scorecard {idx}/10 (Code: {code})...")
        details = parse_scorecard(code)
        if details:
            details['queried_team'] = team_name
            results.append(details)
        time.sleep(0.5) # rate limit politely
        
    return results

def main():
    # Scraping data for India (IND) and Australia (AUS)
    teams_to_scrape = [("IND", "India"), ("AUS", "Australia")]
    all_results = []
    
    for country_code, country_name in teams_to_scrape:
        matches = get_last_10_completed_matches(country_code, country_name)
        all_results.extend(matches)
        
    # Write to CSV file
    csv_file = "match_data.csv"
    fields = ["Match Date", "Team 1", "Team 2", "Venue", "Match Result", "Top Scorer", "Top Score"]
    
    print(f"\nSaving collected data to '{csv_file}'...")
    with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        for row in all_results:
            writer.writerow([
                row["date"],
                row["team1"],
                row["team2"],
                row["venue"],
                row["result"],
                row["top_scorer"],
                row["score"]
            ])
            
    print("Scraping completed successfully! match_data.csv created.")

if __name__ == "__main__":
    main()
