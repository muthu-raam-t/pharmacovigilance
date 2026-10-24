import { render, screen, fireEvent } from '@testing-library/react';
import App from './App';

const mockResult = {
  drug: 'naproxen',
  disease: 'nephrotoxicity',
  suitability: {
    verdict: 'NOT_RECOMMENDED',
    reason: 'Human-annotated literature reports naproxen as a cause of nephrotoxicity.',
    source: 'gold_annotation',
    evidence_type: 'gold',
    model_name: null,
    model_confidence: null,
    has_side_effect_data: true,
    side_effect_sources: ['SIDER', 'OnSIDES'],
    side_effect_count: 2,
    common_side_effect_count: 1,
    boxed_warning_count: 1,
  },
  score_breakdown: [
    { factor: 'Human-annotated adverse-effect evidence', impact: 'negative', weight: 0.9, detail: 'From BC5CDR.' },
  ],
  evidence_sentences: [
    { sentence: 'Naproxen caused nephrotoxicity in two patients.', relation_type: 'CAUSES', origin: 'gold',
      source_dataset: 'BC5CDR', drug_mention: 'Naproxen', disease_mention: 'nephrotoxicity', confidence: null },
  ],
  explanation: 'Naproxen is likely NOT recommended for nephrotoxicity.',
  side_effects: [
    { side_effect: 'gastrointestinal haemorrhage', frequency: 0.12, severity_bucket: 'common', boxed_warning: true, sources: ['SIDER', 'OnSIDES'] },
    { side_effect: 'headache', frequency: null, severity_bucket: 'unknown', boxed_warning: false, sources: ['SIDER'] },
  ],
  total_side_effects: 2,
  alternatives: {
    disease: 'nephrotoxicity', candidates_checked: 1,
    alternatives: [{ drug_name: 'paracetamol', source: 'curated', model_confidence: null, side_effect_count: 10,
      common_side_effect_count: 2, boxed_warning_count: 0, burden_score: 7.5, top_side_effects: ['nausea'] }],
  },
};

afterEach(() => { jest.restoreAllMocks(); });

test('renders the advisory form', () => {
  render(<App />);
  expect(screen.getByRole('heading', { name: /pharmacovigilance advisory/i })).toBeInTheDocument();
  expect(screen.getByLabelText(/drug name/i)).toBeInTheDocument();
  expect(screen.getByLabelText(/disease name/i)).toBeInTheDocument();
});

test('asks for both inputs before calling the API', () => {
  global.fetch = jest.fn();
  render(<App />);
  fireEvent.click(screen.getByRole('button', { name: /^analyze$/i }));
  expect(screen.getByRole('alert')).toHaveTextContent(/both a drug and a disease/i);
  expect(global.fetch).not.toHaveBeenCalled();
});

test('shows verdict, score breakdown, side effects, alternatives and evidence', async () => {
  global.fetch = jest.fn().mockResolvedValue({ ok: true, json: async () => mockResult });
  render(<App />);
  fireEvent.change(screen.getByLabelText(/drug name/i), { target: { value: 'naproxen' } });
  fireEvent.change(screen.getByLabelText(/disease name/i), { target: { value: 'nephrotoxicity' } });
  fireEvent.click(screen.getByRole('button', { name: /^analyze$/i }));

  expect(await screen.findByText(/not recommended/i, { selector: 'h2' })).toBeInTheDocument();
  expect(screen.getByText(/human-annotated adverse-effect evidence/i)).toBeInTheDocument();
  expect(screen.getByText(/gastrointestinal haemorrhage/i)).toBeInTheDocument();
  expect(screen.getByText('Boxed warning')).toBeInTheDocument();
  expect(screen.getByText(/1\. paracetamol/i)).toBeInTheDocument();
  expect(screen.getByText(/in two patients/i)).toBeInTheDocument();
  expect(global.fetch).toHaveBeenCalledWith('http://localhost:8000/analyze', expect.objectContaining({ method: 'POST' }));
});

test('shows a clear message when the API is unreachable', async () => {
  global.fetch = jest.fn().mockRejectedValue(new TypeError('Failed to fetch'));
  render(<App />);
  fireEvent.click(screen.getByText(/metformin \+ type 2 diabetes/i));
  expect(await screen.findByRole('alert')).toHaveTextContent(/cannot reach the api/i);
});
