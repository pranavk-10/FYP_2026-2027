// One place for how each test is named and described in the UI.
// `key` is the upload slot, `resultKey` is the key in the API's individual_cnn_predictions.
export const TESTS = [
  {
    key: 'ecg',
    resultKey: 'ecg',
    title: 'ECG',
    detail: '12-lead printout, scan or photo',
    model: 'ConvNeXt-Tiny v3',
    trainedOn: 'PTB-XL ECG images (GenECG), 5 diagnostic classes',
    accept: 'image/*',
    sample: { path: '/samples/ecg.jpg', name: 'ecg.jpg', type: 'image/jpeg' },
  },
  {
    key: 'xray',
    resultKey: 'xray',
    title: 'Chest X-ray',
    detail: 'PA or AP chest radiograph',
    model: 'DenseNet121',
    trainedOn: 'NIH ChestX-ray14, multi-label',
    accept: 'image/*',
    sample: { path: '/samples/xray.jpeg', name: 'xray.jpeg', type: 'image/jpeg' },
  },
  {
    key: 'skin',
    resultKey: 'dermoscopy',
    title: 'Dermoscopy',
    detail: 'Close-up image of a skin lesion',
    model: 'ResNet18',
    trainedOn: 'HAM10000, 7 lesion classes',
    accept: 'image/*',
    sample: { path: '/samples/skin.jpeg', name: 'skin.jpeg', type: 'image/jpeg' },
  },
  {
    key: 'blood',
    resultKey: 'lab_values',
    title: 'Blood count (CBC)',
    detail: 'Lab report as PDF or photo',
    model: 'OCR + structured extraction',
    trainedOn: 'Tesseract OCR, values checked against reference ranges',
    accept: '.pdf,image/*',
    sample: { path: '/samples/cbc-report-format.pdf', name: 'cbc-report-format.pdf', type: 'application/pdf' },
  },
];

export const EMPTY_SLOTS = { ecg: null, xray: null, skin: null, blood: null };

/** Calibrated probability when available, otherwise the raw one. */
export const probOf = (f) => (f?.calibrated_probability ?? f?.probability ?? 0);

export const pct = (p, digits = 1) => `${(Number(p || 0) * 100).toFixed(digits)}%`;

export const formatBytes = (n) =>
  n < 1024 * 1024 ? `${(n / 1024).toFixed(0)} KB` : `${(n / 1024 / 1024).toFixed(1)} MB`;

export const TIER_LABEL = { detected: 'Detected', watchlist: 'Watch-list', unlikely: 'Unlikely' };
