#!/usr/bin/env python3
"""
Script di test rapido per il sistema three-stage pipeline.
"""

import asyncio
import sys
import logging

# Suppress verbose logging for cleaner output
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

async def test_system():
    print("🔧 Testing Three-Stage Legal NER Pipeline...")

    try:
        print("📦 Importing components...")
        from app.services.ensemble_predictor import EnsemblePredictor

        print("🚀 Initializing EnsemblePredictor...")
        predictor = EnsemblePredictor()
        print("✅ EnsemblePredictor initialized successfully")

        # Test simple legal text
        test_text = "Il decreto legislativo n. 231 del 2001, articolo 25, stabilisce le responsabilità."
        print(f"\n📝 Testing prediction with text:")
        print(f"   '{test_text}'")

        print("\n🔍 Running prediction...")
        entities, requires_review, uncertainty = await predictor.predict(test_text)

        print(f"\n📊 Results:")
        print(f"   • Entities found: {len(entities)}")
        print(f"   • Requires review: {requires_review}")
        print(f"   • Uncertainty: {uncertainty:.3f}")

        if entities:
            print(f"\n🎯 Entities details:")
            for i, entity in enumerate(entities[:3]):  # Show first 3
                print(f"   {i+1}. Text: '{entity.get('text', 'N/A')}'")
                print(f"      Label: {entity.get('label', 'N/A')}")
                print(f"      Confidence: {entity.get('confidence', 0):.3f}")
                if entity.get('structured_data'):
                    print(f"      Structured: {entity['structured_data']}")

        print(f"\n✅ System test completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ System test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_system())
    sys.exit(0 if success else 1)