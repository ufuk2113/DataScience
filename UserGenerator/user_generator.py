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