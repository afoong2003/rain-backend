import requests


def getWeather(lat, lon):
    headers = {
        "User-Agent": "MyRainGardenApp/1.0 afoong2003@gmail.com",
        "Accept": "application/geo+json"
    }

    try:
        points_url = f"https://api.weather.gov/points/{lat},{lon}"
        points_response = requests.get(points_url, headers=headers)
        points_response.raise_for_status()

        points_data = points_response.json()

        stations_url = points_data['properties']['observationStations']
        stations_response = requests.get(stations_url, headers=headers)
        stations_data = stations_response.json()
        station_id = stations_data['features'][0]['properties']['stationIdentifier']

        obs_url = f"https://api.weather.gov/stations/{station_id}/observations/latest"
        weather_response = requests.get(obs_url, headers=headers)
        weather_response.raise_for_status()

        weather_data = weather_response.json()
        properties = weather_data['properties']

        temp_c = properties['temperature']['value']

        if temp_c is not None:
            temp_f = (temp_c * 9/5) + 32
            print(f"\nCurrent Temperature: {temp_f:.1f}°F")
        else:
            print("\nCurrent Temperature: Data unavailable")
        
        return {
            "temperature_f": temp_f if temp_c is not None else None,
        }
    except Exception as err:
        print(err)
    
   
#getWeather(41.832727, -87.642533)
getWeather(41.9769, -87.9081)