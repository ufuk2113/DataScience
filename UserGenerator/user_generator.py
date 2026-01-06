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

    def generate_user_id(self):
        """Erzeugt eindeutige ID zw. 0 und 100."""
        while True:
            uid = random.randint(0, 900000)
            if uid not in self.used_ids:
                self.used_ids.add(uid)
                return uid

    def generate_phone(self):
        """
        Erzeugt deutsche Telefonnummer (z.B. +49 151 23456789)
        Validierung per Regex.
        """
        pattern = r"^\+49\s1[5-7]\d\s\d{7,8}$"
        while True:
            number = f"+49 {self.fake.msisdn()[0:3]} {self.fake.msisdn()[3:11]}"
            if re.match(pattern, number):
                return number

    def generate_income(self):
        """Realistisches monatliches Nettoeinkommen in Deutschland."""
        return random.randint(1200, 6500)

    def generate_birthdate_and_age(self):
        """Geburtsdatum + Alter."""
        birthdate = self.fake.date_of_birth(minimum_age=18, maximum_age=80)
        age = datetime.now().year - birthdate.year
        return birthdate.strftime("%Y-%m-%d"), age

    def generate_german_geolocation(self):
        """
        Liefert reale Stadt-Koordinaten ODER
        zufällige deutsche Koordinaten.
        """
        if random.random() < 0.7:
            city = random.choice(list(self.german_cities.keys()))
            lat, lon = self.german_cities[city]
        else:
            city = self.fake.city()
            lat = random.uniform(47.3, 55.0)
            lon = random.uniform(6.0, 14.5)
        return city, round(lat, 5), round(lon, 5)

    def create_user(self):
        """Erstellt komplettes User-Objekt."""
        birthdate, age = self.generate_birthdate_and_age()
        city, lat, lon = self.generate_german_geolocation()

        return User(
            user_id=self.generate_user_id(),
            name=self.fake.name(),
            birthdate=birthdate,
            age=age,
            address=self.fake.address().replace("\n", ", "),
            email=self.fake.email(),
            phone=self.generate_phone(),
            job=self.fake.job(),
            company=self.fake.company(),
            income=self.generate_income(),
            marital_status=random.choice(self.marital_options),
            household_size=random.randint(1, 5),
            category=random.choice(self.categories),
            city=city,
            latitude=lat,
            longitude=lon
        )

    def create_dataset(self, n: int):
        """Erzeugt DataFrame mit n Nutzern."""
        #users = [self.create_user() for _ in range(n)]
        users = [self.create_user() for _ in tqdm(range(n), desc="Generiere Nutzer")]
        return pd.DataFrame([u.__dict__ for u in users])

    # --------------------- EXPORT FUNKTIONEN ---------------------

    def export_csv(self, df, filename="users.csv"):
        df.to_csv(filename, index=False)

    def export_json(self, df, filename="users.json"):
        df.to_json(filename, orient="records", indent=4)

    def export_xlsx(self, df, filename="users.xlsx"):
        df.to_excel(filename, index=False)

    def export_sql(self, df, filename="users.sql"):
        """
        Exportiert SQL-INSERT-Statements (MariaDB kompatibel).
        """
        table_name = "users"
        sql_lines = [
            f"INSERT INTO {table_name} "
            f"({', '.join(df.columns)}) VALUES"
        ]

        for _, row in df.iterrows():
            values = []
            for value in row:
                if isinstance(value, str):
                    value = value.replace("'", "''")
                    values.append(f"'{value}'")
                else:
                    values.append(str(value))
            sql_lines.append(f"({', '.join(values)}),")

        sql_lines[-1] = sql_lines[-1].rstrip(",")

        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(sql_lines))

        return filename
