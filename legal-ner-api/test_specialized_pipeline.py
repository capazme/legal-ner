#!/usr/bin/env python3
"""
Test Specialized Pipeline
========================

Test completo della pipeline specializzata configurabile per Legal-NER.
"""

import asyncio
import sys
import os
from pathlib import Path

# Aggiungi il path del progetto
sys.path.insert(0, str(Path(__file__).parent))

from app.core.config_loader import get_pipeline_config, ConfigLoader
from app.services.specialized_pipeline import LegalSourceExtractionPipeline
import structlog

# Configura logging per test
structlog.configure(
    processors=[structlog.stdlib.add_log_level, structlog.dev.ConsoleRenderer()],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

log = structlog.get_logger()

async def test_config_loading():
    """Test caricamento configurazione."""
    print("🔧 Testing Configuration Loading...")

    try:
        # Test caricamento configurazione
        config = get_pipeline_config()
        print(f"✅ Configuration loaded successfully")
        print(f"   - Entity detector model: {config.models.entity_detector_primary}")
        print(f"   - Legal classifier model: {config.models.legal_classifier_primary}")
        print(f"   - Rule-based threshold: {config.confidence.rule_based_priority_threshold}")
        print(f"   - NORMATTIVA mappings: {len(config.normattiva_mapping)} categories")

        # Test mappatura flat
        loader = ConfigLoader()
        loader.load_config()
        flat_mapping = loader.get_normattiva_flat_mapping()
        print(f"   - Total abbreviations: {len(flat_mapping)}")

        # Mostra alcuni esempi
        examples = list(flat_mapping.items())[:5]
        print(f"   - Example mappings: {examples}")

        return True

    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

async def test_pipeline_initialization():
    """Test inizializzazione pipeline con configurazione."""
    print("\n🔧 Testing Pipeline Initialization...")

    try:
        # Inizializza pipeline con configurazione
        pipeline = LegalSourceExtractionPipeline()
        print("✅ Pipeline initialized successfully with configuration")

        # Verifica che i componenti abbiano accesso alla configurazione
        print(f"   - EntityDetector NORMATTIVA entries: {len(pipeline.entity_detector.normattiva_mapping)}")
        print(f"   - LegalClassifier has prototypes: {hasattr(pipeline.legal_classifier, 'prototype_embeddings')}")
        print(f"   - Config loaded in all stages: {pipeline.config is not None}")

        return True

    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        return False

async def test_configurable_extraction():
    """Test estrazione con parametri configurabili."""
    print("\n🔧 Testing Configurable Extraction...")

    # Testi di esempio
    test_cases = [
        "Il decreto legislativo n. 231 del 2001 disciplina la responsabilità amministrativa.",
        "Secondo l'art. 5 del c.c., ogni persona ha la capacità giuridica.",
        "La Costituzione italiana all'art. 21 garantisce la libertà di espressione.",
        "Il D.Lgs. 81/2008 sulla sicurezza sul lavoro è fondamentale.",
        "Il DPR 445/2000 disciplina la documentazione amministrativa."
    ]

    try:
        pipeline = LegalSourceExtractionPipeline()
        results_summary = []

        for i, test_text in enumerate(test_cases, 1):
            print(f"\n   Test {i}: {test_text[:50]}...")

            try:
                results = await pipeline.extract_legal_sources(test_text)

                if results:
                    for result in results:
                        print(f"   ✅ Found: {result['text']}")
                        print(f"      Type: {result['act_type']}")
                        print(f"      Confidence: {result['confidence']:.3f}")
                        if result.get('act_number'):
                            print(f"      Number: {result['act_number']}")
                        if result.get('date'):
                            print(f"      Date: {result['date']}")

                    results_summary.append({
                        'test': i,
                        'found': len(results),
                        'text': test_text[:30] + "..."
                    })
                else:
                    print(f"   ⚠️  No results found")
                    results_summary.append({
                        'test': i,
                        'found': 0,
                        'text': test_text[:30] + "..."
                    })

            except Exception as e:
                print(f"   ❌ Error in test {i}: {e}")
                results_summary.append({
                    'test': i,
                    'found': -1,
                    'error': str(e),
                    'text': test_text[:30] + "..."
                })

        # Riepilogo risultati
        print(f"\n📊 Results Summary:")
        total_found = sum(r['found'] for r in results_summary if r['found'] > 0)
        total_tests = len(results_summary)
        success_tests = len([r for r in results_summary if r['found'] > 0])

        print(f"   - Total tests: {total_tests}")
        print(f"   - Successful extractions: {success_tests}")
        print(f"   - Total entities found: {total_found}")
        print(f"   - Success rate: {success_tests/total_tests*100:.1f}%")

        return success_tests > 0

    except Exception as e:
        print(f"❌ Configurable extraction test failed: {e}")
        return False

async def test_confidence_configuration():
    """Test configurazione delle confidence."""
    print("\n🔧 Testing Confidence Configuration...")

    try:
        config = get_pipeline_config()

        # Verifica che le confidence siano nell'intervallo corretto
        confidence_checks = [
            ("minimum_detection_confidence", config.confidence.minimum_detection_confidence),
            ("rule_based_priority_threshold", config.confidence.rule_based_priority_threshold),
            ("specific_codes", config.confidence.specific_codes),
            ("decreto_legislativo_full", config.confidence.decreto_legislativo_full),
            ("legge_full", config.confidence.legge_full),
        ]

        all_valid = True
        for name, value in confidence_checks:
            if 0.0 <= value <= 1.0:
                print(f"   ✅ {name}: {value}")
            else:
                print(f"   ❌ {name}: {value} (out of range [0.0, 1.0])")
                all_valid = False

        # Test che le confidence siano logicamente ordinate
        if config.confidence.specific_codes >= config.confidence.generic_codes:
            print(f"   ✅ Logical ordering: specific_codes >= generic_codes")
        else:
            print(f"   ⚠️  Logical issue: specific_codes < generic_codes")

        if config.confidence.decreto_legislativo_full >= config.confidence.decreto_legislativo_abbrev:
            print(f"   ✅ Logical ordering: full_form >= abbreviation")
        else:
            print(f"   ⚠️  Logical issue: full_form < abbreviation")

        return all_valid

    except Exception as e:
        print(f"❌ Confidence configuration test failed: {e}")
        return False

async def test_context_windows():
    """Test configurazione finestre di contesto."""
    print("\n🔧 Testing Context Windows Configuration...")

    try:
        config = get_pipeline_config()

        # Verifica che le finestre siano positive
        window_checks = [
            ("left_window", config.context.left_window),
            ("right_window", config.context.right_window),
            ("context_window", config.context.context_window),
            ("immediate_context", config.context.immediate_context),
            ("extended_context", config.context.extended_context),
            ("full_context", config.context.full_context),
            ("classification_context", config.context.classification_context),
        ]

        all_valid = True
        for name, value in window_checks:
            if value > 0:
                print(f"   ✅ {name}: {value}")
            else:
                print(f"   ❌ {name}: {value} (must be positive)")
                all_valid = False

        # Test logica delle finestre
        if config.context.extended_context >= config.context.immediate_context:
            print(f"   ✅ Logical ordering: extended >= immediate context")
        else:
            print(f"   ⚠️  Logical issue: extended < immediate context")

        return all_valid

    except Exception as e:
        print(f"❌ Context windows test failed: {e}")
        return False

async def main():
    """Main test execution."""
    print("🚀 Starting Configurable Pipeline Tests\n")

    tests = [
        ("Configuration Loading", test_config_loading),
        ("Pipeline Initialization", test_pipeline_initialization),
        ("Configurable Extraction", test_configurable_extraction),
        ("Confidence Configuration", test_confidence_configuration),
        ("Context Windows", test_context_windows),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)

        try:
            result = await test_func()
            results.append((test_name, result))

            if result:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")

        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
            results.append((test_name, False))

    # Riepilogo finale
    print(f"\n{'='*60}")
    print("📊 FINAL RESULTS")
    print('='*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")

    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("🎉 ALL TESTS PASSED! Configurable pipeline is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the configuration and implementation.")

    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)