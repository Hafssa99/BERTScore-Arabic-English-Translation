"""
BERTScore Evaluation Script
Author: Hafssa Laabdi
Date: 2025-10-26
Description:
Compute BERTScore for Arabic-English metaphor translations
to compare Large Language Models (LLMs) and Neural Machine Translation (NMT) outputs.
"""
from bert_score import score
import numpy as np

# Human references (20 translations)
references = [
    # Extract 1 to 20
    "He dreams of her breast at bloom in evening",
    "His mother’s anguished voice was carving out a new longing under his skin",
    "I am dreaming of white lilies, of a street that is singing",
    "I should breathe in consumption",
    "I must be interned with memories",
    "Why do I smuggle you from airport to airport?",
    "As for my original name, It has been torn from my flesh",
    "Houses die when their inhabitants are gone…",
    "Eternity opens its gates from a distance.",
    "A reckless tomorrow chewed at the wind.",
    "My days hovered over her and before her",
    "And clouds waved to us.",
    "There’s a love walking on two silken feet.",
    "Did you smell the jasmine’s radiant blood?",
    "Whenever a poet dives into himself",
    "She is thrilled by the river, Plunging in rhythm.",
    "But so that she may kill you To avenge her femininity And escape the snare of metaphor.",
    "She walks through the wilds of her self Two unreconciled women",
    "Our desires dried up in us.",
    "No memories Flying from Babylon over our palm tree."
]

# DeepL translations
deepl_translations = [
    "He dreams of her lush bosom in the evening",
    "His mother's anguished voice was digging under his skin a new wish",
    "I dream of white lilies , of a singing street",
    "I have to breathe TB",
    "I should be imprisoned by memories",
    "Why do I smuggle you from airport to airport?",
    "My original name was ripped from my flesh",
    "Houses die when their inhabitants are absent",
    "Eternity opens its doors from afar",
    "It was a wild tomorrow chewing on the wind",
    "My days hover over her... And about her",
    "Clouds waved to us",
    "There is a love that walks on silk feet",
    "Did you smell the blood of the radiant jasmine?",
    "Every time a poet dives into himself",
    "She likes the rhythmic rush of the river",
    "but to kill you to avenge her femininity and the metaphorical trap.",
    "She walks in her own wilderness, two women who don't reconcile.",
    "Our desires have dried up in us.",
    "No memories fly from Babylon over our palm tree"
]

# DeepSeek translations
deepseek_translations = [
    "He dreams of her verdant bosom in the evening.",
    "His mother’s anguished voice carved a new longing beneath his skin.",
    "I dream of white lilies, Of a singing street",
    "I must inhale tuberculosis.",
    "I must be shackled by memories.",
    "Why do I chase you from airport to airport?",
    "As for my true name, It was torn from my flesh",
    "For houses die when their dwellers vanish.",
    "Eternity cracks open its doors from afar.",
    "And there was a reckless tomorrow, chewing the wind.",
    "My days circle around her… and before her.",
    "Clouds waved to us.",
    "There, a love walks on silken feet.",
    "Have you smelled the blood of spilled jasmine?",
    "Whenever a poet drowns in his soul,",
    "She adores the river’s reckless pulse in rhythm.",
    "No—to kill you in vengeance for her femininity, And the trap of metaphor.",
    "She wanders the wilderness of herself: two women refusing truce.",
    "Desires withered within us.",
    "No memories Fly from Babylon’s ruins over our palm tree."
]

# Calculate BERTScore
P_dl, R_dl, F1_dl = score(deepl_translations, references, lang="en", verbose=True)
P_ds, R_ds, F1_ds = score(deepseek_translations, references, lang="en", verbose=True)

# Mean scores
deepl_mean = {
    'Precision': round(P_dl.mean().item(), 4),
    'Recall': round(R_dl.mean().item(), 4),
    'F1': round(F1_dl.mean().item(), 4)
}

deepseek_mean = {
    'Precision': round(P_ds.mean().item(), 4),
    'Recall': round(R_ds.mean().item(), 4),
    'F1': round(F1_ds.mean().item(), 4)
}

# Absolute difference
abs_diff = {
    'Precision': round(abs(deepl_mean['Precision'] - deepseek_mean['Precision']), 4),
    'Recall': round(abs(deepl_mean['Recall'] - deepseek_mean['Recall']), 4),
    'F1': round(abs(deepl_mean['F1'] - deepseek_mean['F1']), 4)
}

# Print results
print("=== DeepL BERTScore ===")
print(deepl_mean)
print("\n=== DeepSeek BERTScore ===")
print(deepseek_mean)
print("\n=== Absolute Difference ===")
print(abs_diff)
