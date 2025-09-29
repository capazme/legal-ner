#!/usr/bin/env python3
"""
Test per la Specialized Legal Source Extraction Pipeline
"""

import asyncio
import sys
import logging

# Suppress verbose logging for cleaner output
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

async def test_specialized_pipeline():
    print("🔧 Testing Specialized Legal Source Extraction Pipeline...")

    try:
        print("📦 Importing specialized pipeline...")
        from app.services.specialized_pipeline import LegalSourceExtractionPipeline

        print("🚀 Initializing Specialized Pipeline...")
        pipeline = LegalSourceExtractionPipeline()
        print("✅ Specialized Pipeline initialized successfully")

        # Test cases con diverse tipologie di riferimenti normativi
        test_cases = [
            {
                "name": "Decreto Legislativo classico",
                "text": "Il decreto legislativo n. 231 del 2001, articolo 25, comma 2, stabilisce le responsabilità amministrative delle società."
            },
            {
                "name": "Abbreviazioni multiple",
                "text": "Il D.Lgs. 81/2008 e il DPR 445/2000 disciplinano la materia. Secondo l'art. 5 del c.c. e l'art. 25 del c.p., sono previste sanzioni."
            },
            {
                "name": "Codici con abbreviazioni",
                "text": "L'articolo 1234 del c.c. stabilisce che, ai sensi dell'art. 589 c.p.c., la procedura deve essere rispettata."
            },
            {
                "name": "Riferimenti incompleti",
                "text": "La legge n. 123/2020 prevede che l'articolo 15, comma 3, sia applicabile. Inoltre, il comma 4 del medesimo articolo dispone diversamente."
            },
            {
                "name": "Costituzione e Codici",
                "text": "L'art. 21 della Costituzione garantisce la libertà di stampa. Il CAD disciplina il processo telematico."
            }
        ]

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🧪 Test Case {i}: {test_case['name']}")
            print(f"📝 Text: \"{test_case['text'][:80]}{'...' if len(test_case['text']) > 80 else ''}\"")

            print("🔍 Running specialized extraction...")
            results = await pipeline.extract_legal_sources(test_case['text'])

            print(f"📊 Results: {len(results)} legal sources detected")

            if results:
                for j, result in enumerate(results, 1):
                    print(f"   {j}. \"{result['text']}\"")
                    print(f"      Position: {result['start_char']}-{result['end_char']}")
                    print(f"      Act Type: {result.get('act_type', 'N/A')}")
                    print(f"      Classification Confidence: {result.get('classification_confidence', 0):.3f}")
                    print(f"      Detection Confidence: {result.get('detection_confidence', 0):.3f}")
                    print(f"      Stage: {result['stage']}")

        print(f"\n✅ All tests completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Specialized pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_specialized_pipeline())
    sys.exit(0 if success else 1)