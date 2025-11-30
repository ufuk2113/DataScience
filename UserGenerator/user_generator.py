import pandas as pd
import random
import re
from faker import Faker
from datetime import datetime
import json
from tqdm import tqdm


class User:
    """
    Repräsentiert einen einzelnen deutschen Nutzer.

    Attribute:
        user_id (int): Eindeutige ID (0–900000)
        name (str): Vollständiger Name
        birthdate (str): Geburtsdatum im Format YYYY-MM-DD
        age (int): Alter in Jahren
        address (str): Deutsche Adresse
        email (str): E-Mail-Adresse
        phone (str): Deutsche Telefonnummer
        job (str): Beruf
        company (str): Arbeitgeber
        income (int): Monatliches Nettoeinkommen in €
        marital_status (str): Ledig / Verheiratet / Geschieden / Verwitwet
        household_size (int): Anzahl Haushaltsmitglieder
        category (str): Nutzerkategorie (Basic, Premium, Business, VIP)
        city (str): Deutscher Ort
        latitude (float): Breitengrad (Deutschland)
        longitude (float): Längengrad (Deutschland)
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
class UserGenerator:
    """
    Erzeugt realistische deutsche Nutzerdaten mit erweiterten Profilen.

    Funktionen:
        - generate_user_id(): Eindeutige ID
        - generate_phone(): Deutsche Telefonnummer via Regex
        - generate_income(): Deutsches Netto-Einkommen
        - generate_birthdate_and_age(): Geburtsdatum + Altersberechnung
        - generate_german_geolocation(): Koordinaten für DE (oder reale Stadt-Koordinaten)
        - create_user(): Erstellt komplettes Nutzerobjekt
        - create_dataset(n): Erstellt DataFrame mit n Nutzern
        - export_csv/json/xlsx/sql(): Verschiedene Exporte
    """

    # Realistische deutsche Stadt-Koordinaten
    german_cities = {
        "Berlin": (52.5200, 13.4050),
        "Hamburg": (53.5511, 9.9937),
        "München": (48.1351, 11.5820),
        "Köln": (50.9375, 6.9603),
        "Frankfurt am Main": (50.1109, 8.6821),
        "Stuttgart": (48.7758, 9.1829),
        "Düsseldorf": (51.2277, 6.7735),
        "Leipzig": (51.3397, 12.3731),
        "Dortmund": (51.5136, 7.4653)
    }

    marital_options = ["Ledig", "Verheiratet", "Geschieden", "Verwitwet"]
    categories = ["Basic", "Premium", "Business", "VIP"]

    def __init__(self):
        self.used_ids = set()
        self.fake = Faker("de_DE")
        