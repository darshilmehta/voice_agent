/**
 * The product's name and tagline (docs/DESIGN.md, decided 2026-10-10). The name is Docent in every language (Latin
 * script even in Hindi text); the tagline has a Hindi form for Hindi UI text.
 *
 * The browser shows the name the backend sends (`client.app_title` in the config, GET /api/config/public); this is the
 * fallback while that loads or when the backend is down, and the tagline's home.
 */

export const PRODUCT_NAME = "Docent";
export const TAGLINE = "Talk to your documents";
export const TAGLINE_HI = "अपने दस्तावेज़ों से बात करें";
