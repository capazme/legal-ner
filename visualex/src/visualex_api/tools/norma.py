from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
import structlog

# Assuming these functions are defined elsewhere in your code
from .urngenerator import generate_urn
from .text_op import normalize_act_type
from .treextractor import get_tree
from .config import MAX_CACHE_SIZE

log = structlog.get_logger()

@dataclass
class Norma:
    tipo_atto: str
    data: str = None
    numero_atto: str = None
    _url: str = None
    _tree: any = field(default=None, repr=False)

    def __post_init__(self):
        log.debug(f"Initializing Norma with tipo_atto: {self.tipo_atto}, data: {self.data}, numero_atto: {self.numero_atto}")
        self.tipo_atto_str = normalize_act_type(self.tipo_atto, search=True)
        self.tipo_atto_urn = normalize_act_type(self.tipo_atto)
        log.debug(f"Norma initialized: {self}")

    @property

    def url(self):
        if not self._url:
            log.debug("Generating URL for Norma.")
            generated_urn = generate_urn(
                act_type=self.tipo_atto_urn,
                date=self.data,
                act_number=self.numero_atto,
                urn_flag=False
            )
            if generated_urn is None:
                log.error(f"Failed to generate URN for {self.tipo_atto_urn} {self.data} {self.numero_atto}")
                raise ValueError(f"Cannot generate URL for legal norm: {self}")
            self._url = generated_urn
        return self._url

    @property
     
    def tree(self):
        if not self._tree:
            log.debug("Fetching tree structure for Norma.")
            self._tree = get_tree(self.url)
        return self._tree

    def __str__(self):
        parts = [self.tipo_atto_str]
        if self.data:
            parts.append(f"{self.data},")
        if self.numero_atto:
            parts.append(f"n. {self.numero_atto}")
        return " ".join(parts)

    def to_dict(self):
        return {
            'tipo_atto': self.tipo_atto_str,
            'data': self.data,
            'numero_atto': self.numero_atto,
            'url': self.url,
        }

@dataclass(eq=False)
class NormaVisitata:
    norma: Norma
    allegato: str = None
    numero_articolo: str = None
    versione: str = None
    data_versione: str = None
    _urn: str = field(default=None, repr=False)
    #timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __hash__(self):
        return hash((self.norma.tipo_atto_urn, self.norma.data, self.norma.numero_atto, self.numero_articolo, self.versione, self.data_versione))

    def __eq__(self, other):
        if not isinstance(other, NormaVisitata):
            return NotImplemented
        return (self.norma.tipo_atto_urn == other.norma.tipo_atto_urn and
                self.norma.data == other.norma.data and
                self.norma.numero_atto == other.norma.numero_atto and
                self.numero_articolo == other.numero_articolo and
                self.versione == other.versione and
                self.data_versione == other.data_versione and self.allegato == other.allegato)

    def __post_init__(self):
        log.debug(f"NormaVisitata initialized: {self}")

    @property

    def urn(self):
        if not self._urn:
            log.debug("Generating URN for NormaVisitata.")
            generated_urn = generate_urn(
                act_type=self.norma.tipo_atto_urn,
                date=self.norma.data,
                act_number=self.norma.numero_atto,
                annex = self.allegato,
                article=self.numero_articolo,
                version=self.versione,
                version_date=self.data_versione
            )
            if generated_urn is None:
                log.error(f"Failed to generate URN for {self.norma}")
                raise ValueError(f"Cannot generate URN for visited norm: {self}")
            self._urn = generated_urn
        return self._urn

    def __str__(self):
        base_str = str(self.norma)
        if self.numero_articolo:
            base_str += f" art. {self.numero_articolo}"
        return base_str

    def to_dict(self):
        base_dict = self.norma.to_dict()
        base_dict.update({
            'allegato': self.allegato,
            'numero_articolo': self.numero_articolo,
            'versione': self.versione,
            'data_versione': self.data_versione,
            #'timestamp': self.timestamp,
            'urn': self.urn,
        })
        return base_dict

    @staticmethod
    def from_dict(data):
        log.debug(f"Creating NormaVisitata from dict: {data}")
        norma = Norma(
            tipo_atto=data['tipo_atto'],
            data=data.get('data'),
            numero_atto=data.get('numero_atto'),
            _url=data.get('url'),
            _tree=data.get('tree')
        )
        norma_visitata = NormaVisitata(
            norma=norma,
            numero_articolo=data.get('numero_articolo'),
            versione=data.get('versione'),
            data_versione=data.get('data_versione'),
            allegato = data.get('allegato')
            #timestamp=data.get('timestamp')
        )
        log.debug(f"NormaVisitata created: {norma_visitata}")
        return norma_visitata

codice_civile = Norma(tipo_atto='codice civile')
print(codice_civile.to_dict())