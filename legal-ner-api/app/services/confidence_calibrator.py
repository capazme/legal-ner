"""
RESET - Confidence Calibrator Service
======================================

Questo modulo è stato completamente resettato. Contiene solo l'interfaccia base
per mantenere la compatibilità con l'architettura esistente.

NOTA: Tutta la logica di calibrazione multi-fattore è stata rimossa
per permettere un ripensamento completo dell'approccio.
"""

from typing import List, Dict, Any
import structlog

log = structlog.get_logger()

class ConfidenceCalibrator:
    """
    Interfaccia base per il calibratore di confidenza.

    STATO: RESET COMPLETO
    - Rimossa logica di calibrazione multi-fattore precedente
    - Rimossi parametri di calibrazione e normalizzazione
    - Mantenuta solo l'interfaccia per compatibilità architetturale
    - Pronto per riprogettazione completa
    """

    def __init__(self):
        log.info("ConfidenceCalibrator RESET - Inizializzazione interfaccia base")
        # TODO: Riprogettare completamente l'approccio alla calibrazione
        pass

    def calibrate(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Interfaccia base per calibrazione confidenza.

        Args:
            entities: Lista di entità con confidence scores

        Returns:
            Lista di entità con confidence calibrate

        STATO: PLACEHOLDER - Implementazione da riprogettare
        """
        log.warning("CALIBRATE chiamato su servizio RESET - implementazione placeholder")

        # Placeholder: pass-through senza modifiche durante reset
        return entities