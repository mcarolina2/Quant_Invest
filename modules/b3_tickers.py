B3_TICKERS_BY_SECTOR: dict[str, list[str]] = {
    "🛢️ Petróleo & Gás": [
        "PETR3.SA","PETR4.SA","PRIO3.SA","RECV3.SA","RRRP3.SA",
        "CSAN3.SA","UGPA3.SA","VBBR3.SA",
    ],
    "⛏️ Mineração & Siderurgia": [
        "VALE3.SA","CSNA3.SA","GGBR3.SA","GGBR4.SA",
        "USIM3.SA","USIM5.SA","GOAU3.SA","GOAU4.SA",
        "FESA4.SA","CMIN3.SA","CBAV3.SA",
    ],
    "🏦 Bancos & Financeiro": [
        "ITUB3.SA","ITUB4.SA","BBDC3.SA","BBDC4.SA","BBAS3.SA",
        "SANB3.SA","SANB4.SA","SANB11.SA","BPAC11.SA","BPAN4.SA",
        "BRSR3.SA","BRSR6.SA","BMGB4.SA","ABCB4.SA","PINE4.SA",
        "IRBR3.SA","SULA11.SA","BBSE3.SA","PSSA3.SA","CXSE3.SA",
        "WIZC3.SA","CVCB3.SA",
    ],
    "🛍️ Varejo": [
        "MGLU3.SA","VIIA3.SA","AMER3.SA","LREN3.SA","ALPA4.SA",
        "AMAR3.SA","CEAB3.SA","SOMA3.SA","GRND3.SA","VIVA3.SA",
        "SBFG3.SA","SMFT3.SA","TFCO4.SA","MOVI3.SA",
    ],
    "🍺 Consumo & Alimentos": [
        "ABEV3.SA","BRFS3.SA","JBSS3.SA","MRFG3.SA","BEEF3.SA",
        "MDIA3.SA","PCAR3.SA","ASAI3.SA","GMAT3.SA","CRFB3.SA",
    ],
    "🏥 Saúde": [
        "RDOR3.SA","HAPV3.SA","FLRY3.SA","DASA3.SA","PARD3.SA",
        "ONCO3.SA","MATD3.SA","BLAU3.SA",
    ],
    "⚡ Energia Elétrica": [
        "ELET3.SA","ELET6.SA","CPFE3.SA","CMIG3.SA","CMIG4.SA",
        "EGIE3.SA","ENGI3.SA","ENGI11.SA","EQTL3.SA","AURE3.SA",
        "CESP3.SA","CESP6.SA","ENBR3.SA","ENEV3.SA","CPLE3.SA",
        "CPLE6.SA","AESB3.SA","COCE5.SA","LIGT3.SA","NEOE3.SA",
        "TIET3.SA","TIET11.SA",
    ],
    "📡 Telecomunicações": [
        "VIVT3.SA","TIMS3.SA","OIBR3.SA","OIBR4.SA",
    ],
    "🏗️ Construção Civil": [
        "MRVE3.SA","CYRE3.SA","EVEN3.SA","EZTC3.SA","TEND3.SA",
        "DIRR3.SA","PLPL3.SA","JHSF3.SA","HBOR3.SA","TRIS3.SA",
        "GFSA3.SA",
    ],
    "🚛 Logística & Transporte": [
        "RAIL3.SA","CCRO3.SA","ECOR3.SA","AZUL4.SA","GOLL4.SA",
        "EMBR3.SA","MILS3.SA","RLOG3.SA","POMO3.SA","POMO4.SA",
        "TGMA3.SA",
    ],
    "🌾 Agronegócio": [
        "SLCE3.SA","AGRO3.SA","SMTO3.SA","TTEN3.SA","CAML3.SA",
    ],
    "📄 Papel & Celulose": [
        "SUZB3.SA","KLBN3.SA","KLBN4.SA","KLBN11.SA",
    ],
    "🧪 Químico & Petroquímico": [
        "BRKM3.SA","BRKM5.SA","UNIP3.SA","UNIP6.SA",
    ],
    "💻 Tecnologia": [
        "TOTVS3.SA","LWSA3.SA","CASH3.SA","IFCM3.SA","DESK3.SA",
        "INTB3.SA",
    ],
    "🏬 Shoppings & Imobiliário": [
        "MULT3.SA","IGTI3.SA","IGTI11.SA","ALLD3.SA",
    ],
    "💧 Saneamento": [
        "SAPR3.SA","SAPR4.SA","SAPR11.SA","SBSP3.SA","CSMG3.SA",
    ],
    "🏭 Indústria Diversificada": [
        "WEGE3.SA","RENT3.SA","RADL3.SA","NTCO3.SA","QUAL3.SA",
        "HYPE3.SA","LEVE3.SA","RANI3.SA","PGMN3.SA",
    ],
    "📦 ETFs B3": [
        "BOVA11.SA","SMAL11.SA","IVVB11.SA","SPXI11.SA",
        "HASH11.SA","GOLD11.SA","DIVO11.SA","XFIX11.SA",
    ],
}

# Lista plana ordenada — usada no multiselect
ALL_TICKERS: list[str] = sorted({
    ticker
    for tickers in B3_TICKERS_BY_SECTOR.values()
    for ticker in tickers
})


def get_all_tickers() -> list[str]:
    """Retorna todos os tickers em ordem alfabética."""
    return ALL_TICKERS


def get_tickers_by_sector(sector: str) -> list[str]:
    """Retorna os tickers de um setor específico."""
    return B3_TICKERS_BY_SECTOR.get(sector, [])


def get_sectors() -> list[str]:
    """Retorna lista de setores disponíveis."""
    return list(B3_TICKERS_BY_SECTOR.keys())


def search_tickers(query: str) -> list[str]:
    """Filtra tickers que contêm a query (case-insensitive)."""
    q = query.upper().replace(".SA", "")
    return [t for t in ALL_TICKERS if q in t.replace(".SA", "")]