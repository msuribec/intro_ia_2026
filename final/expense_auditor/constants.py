"""Shared constants for the My-Expense Auditor app."""

SPENDING_EXPORT_COLUMNS = [
    "receipt_index",
    "vendor",
    "date",
    "currency",
    "item",
    "price",
    "category",
    "receipt_total",
]

DEFAULT_CATEGORIES = [
    "Food & Groceries",
    "Transport",
    "Entertainment",
    "Health & Beauty",
    "Household",
    "Clothing",
    "Utilities",
    "Other",
]

GTTS_SUPPORTED_LANGS = {
    "af", "sq", "ar", "hy", "bn", "bs", "ca", "hr", "cs", "da", "nl",
    "en", "eo", "et", "tl", "fi", "fr", "de", "el", "gu", "hi", "hu",
    "is", "id", "it", "ja", "jw", "kn", "km", "ko", "la", "lv", "mk",
    "ml", "mr", "my", "ne", "no", "pl", "pt", "ro", "ru", "sr", "si",
    "sk", "es", "su", "sw", "sv", "ta", "te", "th", "tr", "uk", "ur",
    "vi", "cy", "zh-cn", "zh-tw",
}

PAGE_TITLE = "My-Expense Auditor"
PAGE_ICON = "\U0001f4b0"
PAGE_LAYOUT = "centered"

