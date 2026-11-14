"""
SoilDoctor - Database Layer (SQLite)
Manages:
- Scan history records
- Farm field geometry & status for Leaflet.js
- Crop Doctor encyclopedia data (Exclusively: Tomato, Rice, Chilli, Sugarcane, Blackgram, Soybean)
- User profile & preferences
"""

import json
import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "soildoctor.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(force_reseed=False):
    conn = get_db()
    cursor = conn.cursor()

    # 1. Scans Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS scans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scan_type TEXT NOT NULL, -- 'plant' or 'soil'
        crop_name TEXT NOT NULL,
        disease_name TEXT NOT NULL,
        confidence INTEGER NOT NULL,
        risk_level TEXT NOT NULL, -- 'LOW', 'MEDIUM', 'HIGH'
        created_at TEXT NOT NULL,
        details_json TEXT NOT NULL,
        thumbnail_b64 TEXT,
        field_id INTEGER,
        location_type TEXT,
        location_lat REAL,
        location_lng REAL
    )
    """)

    # Add scan location metadata to databases created by earlier versions.
    scan_columns = {row[1] for row in cursor.execute("PRAGMA table_info(scans)").fetchall()}
    for column, sql_type in (("field_id", "INTEGER"), ("location_type", "TEXT"),
                             ("location_lat", "REAL"), ("location_lng", "REAL")):
        if column not in scan_columns:
            cursor.execute(f"ALTER TABLE scans ADD COLUMN {column} {sql_type}")

    # 2. Fields Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS fields (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        crop TEXT NOT NULL,
        area_acres REAL NOT NULL,
        status TEXT NOT NULL, -- 'Healthy' or 'Needs Attention'
        last_scan_date TEXT NOT NULL,
        lat REAL NOT NULL,
        lng REAL NOT NULL,
        polygon_json TEXT NOT NULL
    )
    """)

    # 3. Crops Knowledge Base Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS crops (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        icon TEXT NOT NULL,
        category TEXT NOT NULL,
        soil_pref TEXT NOT NULL,
        ph_range TEXT NOT NULL,
        diseases_json TEXT NOT NULL,
        nutrient_guide_json TEXT NOT NULL
    )
    """)

    # 4. User Profile & Settings Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS profile (
        id INTEGER PRIMARY KEY,
        full_name TEXT NOT NULL,
        email TEXT NOT NULL,
        phone TEXT NOT NULL,
        farm_location TEXT NOT NULL,
        notifications INTEGER DEFAULT 1,
        email_updates INTEGER DEFAULT 1,
        dark_mode INTEGER DEFAULT 0
    )
    """)

    conn.commit()

    if force_reseed:
        cursor.execute("DELETE FROM crops")
        cursor.execute("DELETE FROM fields")
        cursor.execute("DELETE FROM scans")
        seed_data(cursor, conn)
    else:
        cursor.execute("SELECT COUNT(*) FROM crops")
        if cursor.fetchone()[0] == 0:
            seed_data(cursor, conn)

    conn.close()

def seed_data(cursor, conn):
    print("Seeding initial database records for Dataset new crops...")
    # Profile
    cursor.execute("DELETE FROM profile")
    cursor.execute("""
    INSERT INTO profile (id, full_name, email, phone, farm_location, notifications, email_updates, dark_mode)
    VALUES (1, 'Ramesh Kumar', 'ramesh@example.com', '+91 98765 43210', 'Coimbatore, Tamil Nadu', 1, 1, 0)
    """)

    # Fields for Leaflet Farm Map (Coordinates around Coimbatore farm: 11.0168° N, 76.9558° E)
    fields = [
        (
            "Plot 1 - North Paddy", "Rice", 4.2, "Healthy", "Today", 11.0190, 76.9520,
            json.dumps([[11.0180, 76.9500], [11.0210, 76.9510], [11.0200, 76.9540], [11.0170, 76.9530]])
        ),
        (
            "Plot 2 - Ridge Farm", "Tomato", 3.5, "Needs Attention", "Today", 11.0225, 76.9575,
            json.dumps([[11.0210, 76.9550], [11.0245, 76.9560], [11.0240, 76.9605], [11.0205, 76.9595]])
        ),
        (
            "Plot 3 - South Orchard", "Chilli", 2.8, "Needs Attention", "Yesterday", 11.0150, 76.9580,
            json.dumps([[11.0135, 76.9565], [11.0165, 76.9570], [11.0160, 76.9610], [11.0130, 76.9600]])
        ),
        (
            "Plot 4 - River Basin", "Sugarcane", 5.0, "Healthy", "3 days ago", 11.0250, 76.9500,
            json.dumps([[11.0235, 76.9480], [11.0270, 76.9490], [11.0265, 76.9530], [11.0230, 76.9520]])
        ),
        (
            "Plot 5 - Agroforestry", "Blackgram", 2.0, "Healthy", "4 days ago", 11.0120, 76.9510,
            json.dumps([[11.0105, 76.9490], [11.0135, 76.9500], [11.0130, 76.9540], [11.0100, 76.9530]])
        )
    ]
    cursor.executemany("""
    INSERT INTO fields (name, crop, area_acres, status, last_scan_date, lat, lng, polygon_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, fields)

    # Initial Scans reflecting Dataset new
    scans = [
        (
            "plant", "Tomato", "Tomato Early Blight", 93, "HIGH", "Today",
            json.dumps({
                "affected_area": "9.4%",
                "symptoms": ["Concentric ring lesions", "Chlorotic halo"],
                "focus": "Copper fungicide spray & moisture control"
            }),
            ""
        ),
        (
            "plant", "Rice", "Rice Healthy Leaf", 96, "LOW", "Yesterday",
            json.dumps({
                "affected_area": "0.3%",
                "symptoms": ["Vibrant green blade", "Uniform pigmentation"],
                "focus": "Routine maintenance"
            }),
            ""
        ),
        (
            "plant", "Chilli", "Chilli Mites & Thrips", 89, "HIGH", "2 days ago",
            json.dumps({
                "affected_area": "6.8%",
                "symptoms": ["Upward and downward curling", "Thickened rugose leaf"],
                "focus": "Neem oil & organic acaricide spray"
            }),
            ""
        ),
        (
            "plant", "Sugarcane", "Sugarcane Red Rot", 91, "HIGH", "3 days ago",
            json.dumps({
                "affected_area": "12.2%",
                "symptoms": ["Reddish lesions on midrib with white cross bands"],
                "focus": "Carbendazim spray & drainage improvement"
            }),
            ""
        ),
        (
            "plant", "Blackgram", "Blackgram Yellow Mosaic", 94, "HIGH", "5 days ago",
            json.dumps({
                "affected_area": "14.5%",
                "symptoms": ["Bright yellow patches on foliage", "Whitefly vector transmission"],
                "focus": "Whitefly control & yellow sticky traps"
            }),
            ""
        ),
        (
            "plant", "Soybean", "Soybean Healthy Leaf", 97, "LOW", "6 days ago",
            json.dumps({
                "affected_area": "0.5%",
                "symptoms": ["Healthy trifoliate leaves", "Clean lamina"],
                "focus": "Maintain balanced fertility"
            }),
            ""
        )
    ]
    cursor.executemany("""
    INSERT INTO scans (scan_type, crop_name, disease_name, confidence, risk_level, created_at, details_json, thumbnail_b64)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, scans)

    # 6 Crops exclusively from Dataset new
    crops = [
        (
            "Tomato", "🍅", "Solanaceae / Vegetable", "Well-drained sandy loam rich in organic matter", "6.0 - 6.8",
            json.dumps([
                {"name": "Early Blight (Alternaria solani)", "symptoms": "Concentric dark brown spots with yellow halos on older leaves.", "treatment": "Copper fungicide, Mancozeb 75% WP, prune lower diseased foliage."},
                {"name": "Late Blight (Phytophthora infestans)", "symptoms": "Rapidly enlarging water-soaked dark lesions turning brown with white sporulation.", "treatment": "Azoxystrobin or Cymoxanil, remove infected tissue immediately."},
                {"name": "Leaf Miner (Liriomyza trifolii)", "symptoms": "Serpentine winding translucent white trails through leaf tissue.", "treatment": "Neem seed kernel extract (NSKE 5%) or Spinosad spray."},
                {"name": "Magnesium Deficiency", "symptoms": "Interveinal yellowing (chlorosis) between green veins on mature leaves.", "treatment": "Foliar spray of 1% Magnesium Sulphate (Epsom salt) solution."},
                {"name": "Nitrogen Deficiency", "symptoms": "Overall pale green to chlorotic yellow foliage with stunted vegetative vigor.", "treatment": "Side-dress with vermicompost or apply 1% urea foliar spray."},
                {"name": "Potassium Deficiency", "symptoms": "Marginal scorching, browning, and upward curling of leaflet tips.", "treatment": "Apply Muriate of Potash (MOP) or sulphate of potash."},
                {"name": "Spotted Wilt Virus (TSWV)", "symptoms": "Bronze specks, circular dark rings, leaf cupping and terminal dieback.", "treatment": "Control thrips vector with sticky traps; rogue out diseased plants."}
            ]),
            json.dumps({
                "nitrogen": "High requirement in early vegetative stage; avoid excess before flowering to prevent blossom drop.",
                "phosphorus": "Essential for early root establishment and blossom cluster formation.",
                "potassium": "Vital for fruit size, lycopene pigment synthesis, and firmness.",
                "magnesium": "Essential central component of chlorophyll molecule."
            })
        ),
        (
            "Rice", "🌾", "Poaceae / Cereal", "Clayey or heavy loams capable of prolonged moisture retention", "5.5 - 6.5",
            json.dumps([
                {"name": "Bacterial Leaf Blight (Xanthomonas oryzae)", "symptoms": "Yellowish-white wavy lesions starting at leaf tips and margins; bacterial ooze.", "treatment": "Bleaching powder drench (2 kg/acre); avoid high nitrogen doses."},
                {"name": "Brown Spot (Bipolaris oryzae)", "symptoms": "Small oval to circular dark brown spots with grey center and yellow halo.", "treatment": "Mancozeb or Propiconazole foliar spray; correct soil potassium."},
                {"name": "Leaf Blast (Magnaporthe oryzae)", "symptoms": "Spindle-shaped diamond lesions with ash-grey center and dark brown border.", "treatment": "Tricyclazole 75% WP or Isoprothiolane foliar spray."},
                {"name": "Leaf Scald (Microdochium oryzae)", "symptoms": "Zonate chevron-like olive-green to bleached bands expanding from tip.", "treatment": "Spray carbendazim + mancozeb combination."},
                {"name": "Narrow Brown Leaf Spot (Cercospora)", "symptoms": "Narrow short reddish-brown linear streaks parallel to leaf veins.", "treatment": "Propiconazole or azoxystrobin spray."},
                {"name": "Rice Hispa (Dicladispa armigera)", "symptoms": "Parallel white streaks scraped on upper surface; blister-like larvae mines.", "treatment": "Clip affected leaf tips; spray chlorpyrifos or neem extract."},
                {"name": "Sheath Blight (Rhizoctonia solani)", "symptoms": "Oval greyish-green water-soaked lesions on leaf sheaths near water level.", "treatment": "Hexaconazole 5% EC or Validamycin foliar application."}
            ]),
            json.dumps({
                "nitrogen": "Split into 3 stages: 50% basal at puddling, 25% at tillering, 25% at panicle initiation.",
                "phosphorus": "Apply 100% as basal dose during final puddling.",
                "potassium": "Provides stalk strength against lodging and disease immunity.",
                "zinc": "Crucial in submerged soils; zinc deficiency causes Khaira rust disease."
            })
        ),
        (
            "Chilli", "🌶️", "Solanaceae / Spice", "Fertile, aerated sandy loam with excellent organic drainage", "6.2 - 7.2",
            json.dumps([
                {"name": "Cercospora Leaf Spot (Frogeye)", "symptoms": "Circular spots with pale ash-grey center and dark reddish-brown margins.", "treatment": "Spray Difenoconazole or Mancozeb 75 WP."},
                {"name": "Mites & Thrips Infestation", "symptoms": "Leaves curl upward like cups (thrips) or downward like boats (mites); bronzed leathery texture.", "treatment": "Spray cold-pressed neem oil (5ml/L) and Fenpyroximate / Spiromesifen for mites."},
                {"name": "Nutritional Deficiency", "symptoms": "Pale chlorotic yellowing, reduced fruit count, and leaf margin yellowing.", "treatment": "Balanced NPK 19-19-19 foliar spray plus chelated micronutrient mix."},
                {"name": "Powdery Mildew (Leveillula taurica)", "symptoms": "White flour-like powdery fungal growth on lower surface; yellow patches above.", "treatment": "Wettable sulfur (2.5g/L) or Dinocap spray."}
            ]),
            json.dumps({
                "nitrogen": "Apply in splits after initial seedling establishment and after first flower flush.",
                "phosphorus": "Essential for rapid taproot branching and uniform flowering set.",
                "potassium": "Directly elevates fruit pungency (capsaicin), deep red color, and shelf-life.",
                "calcium": "Deficiency results in blossom end rot and blossom drop."
            })
        ),
        (
            "Sugarcane", "🎋", "Poaceae / Cash Crop", "Deep, well-drained fertile loams or alluvial soils", "6.5 - 7.5",
            json.dumps([
                {"name": "Sugarcane Mosaic Virus (SCMV)", "symptoms": "Contrasting light and dark green mottled patches; chlorotic linear streaks on leaves.", "treatment": "Plant certified virus-free seed setts; eradicate aphid vectors; rogue infected stools."},
                {"name": "Sugarcane Red Rot (Colletotrichum)", "symptoms": "Reddish spots on leaf midribs with white cross-bands; internal cane red discoloration.", "treatment": "Sett treatment with Carbendazim (1g/L); clean field sanitation and drainage."},
                {"name": "Sugarcane Rust (Puccinia)", "symptoms": "Tiny elongated yellowish spots turning into reddish-brown powdery spore pustules on underside.", "treatment": "Foliar spray of Mancozeb or Propiconazole at onset of rust spots."},
                {"name": "Sugarcane Yellow Leaf Disease (SCYLV)", "symptoms": "Prominent yellowing of midrib on 3rd-6th leaves spreading into leaf blade tissue.", "treatment": "Eradicate aphid vectors (Melanaphis sacchari); apply balanced potassium and silicon."}
            ]),
            json.dumps({
                "nitrogen": "High nitrogen demand during tillering and formative stage (up to 120-150 days).",
                "phosphorus": "Apply full dose as basal in furrows at planting.",
                "potassium": "Essential for sucrose synthesis, translocating sugar to stalks, and drought tolerance."
            })
        ),
        (
            "Blackgram", "🌱", "Fabaceae / Pulse & Legume", "Well-drained loam or clay-loam with moderate moisture retention", "6.5 - 7.5",
            json.dumps([
                {"name": "Anthracnose (Colletotrichum)", "symptoms": "Circular to angular dark brown sunken spots on leaves, petioles, and pods.", "treatment": "Seed treatment with Carbendazim; foliar spray of Mancozeb (2g/L)."},
                {"name": "Leaf Crinkle Virus (ULCV)", "symptoms": "Extreme rugosity, leathery thickening, crinkling and curling of trifoliate leaves.", "treatment": "Use disease-free certified seeds; manage whitefly and aphid vectors."},
                {"name": "Powdery Mildew (Erysiphe polygoni)", "symptoms": "White powdery fungal coating spreading rapidly over foliage.", "treatment": "Spray Wettable Sulfur (3g/L) or Hexaconazole (1ml/L)."},
                {"name": "Yellow Mosaic Virus (MYMV)", "symptoms": "Scattered bright golden-yellow patches intermingled with green tissue on leaves.", "treatment": "Spray systemic insecticide (Dimethoate / Imidacloprid) to eliminate whitefly vector; install yellow sticky traps."}
            ]),
            json.dumps({
                "nitrogen": "Legume crop with Rhizobium root nodules; requires small starter nitrogen only (15-20 kg/acre).",
                "phosphorus": "High phosphorus requirement for vigorous nodulation and ATP energy transfer.",
                "potassium": "Enhances seed grain filling, protein quality, and pod firmness.",
                "rhizobium": "Inoculate seeds with Rhizobium bio-fertilizer culture before planting."
            })
        ),
        (
            "Soybean", "🌿", "Fabaceae / Oilseed & Pulse", "Deep, fertile, well-aerated sandy loam to clay loam", "6.0 - 7.0",
            json.dumps([
                {"name": "Soybean Foliar Disease Complex", "symptoms": "Brown necrotic spots, bacterial pustules, or frog-eye circular lesions across leaf surface.", "treatment": "Foliar application of Pyraclostrobin or Azoxystrobin + Mancozeb; practice crop rotation."},
                {"name": "Soybean Rust (Phakopsora pachyrhizi)", "symptoms": "Small brown-to-tan polygonal pustules on lower leaf surfaces; rapid premature defoliation.", "treatment": "Preventive spray of Triazole fungicides (Propiconazole / Tebuconazole)."}
            ]),
            json.dumps({
                "nitrogen": "Relies heavily on biological nitrogen fixation via Bradyrhizobium japonicum root nodules.",
                "phosphorus": "Critical for early root development, pod setting, and oil synthesis.",
                "potassium": "Improves water use efficiency, disease defense, and seed quality.",
                "sulfur": "Essential for synthesizing methionine and cysteine amino acids and boosting oil content."
            })
        )
    ]
    cursor.executemany("""
    INSERT INTO crops (name, icon, category, soil_pref, ph_range, diseases_json, nutrient_guide_json)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, crops)

    conn.commit()

if __name__ == "__main__":
    init_db(force_reseed=True)
    print("Database re-seeded with Dataset new crops!")
