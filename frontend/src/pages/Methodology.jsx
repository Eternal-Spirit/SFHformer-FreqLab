import { PROJECT_META, HYPOTHESES } from '../data/experimentData';
import { BookOpen, GitBranch, Database, Layers, TestTube, Shield } from 'lucide-react';

function Section({ icon: Icon, title, children }) {
  return (
    <div className="methodology-section">
      <h2 className="methodology-section__title">
        <Icon size={20} /> {title}
      </h2>
      {children}
    </div>
  );
}

export default function Methodology() {
  return (
    <div className="page">
      <div className="page__header">
        <div>
          <h1 className="page__title">Methodology</h1>
          <p className="page__description">
            Experimental design, data pipeline, and pre-registered hypothesis framework
          </p>
        </div>
      </div>

      <Section icon={BookOpen} title="Research Question">
        <div className="methodology-card">
          <blockquote className="methodology-quote">
            "{PROJECT_META.question}"
          </blockquote>
          <p className="methodology-text">
            This experiment investigates the frequency-domain robustness of SFHformer, a hybrid
            spatial-frequency image restoration network published at ECCV 2024. We train on
            band-limited sinusoidal noise (k ∈ [8, 38] cycles/image) and test generalization
            across the full frequency spectrum.
          </p>
        </div>
      </Section>

      <Section icon={GitBranch} title="Pre-Registration Protocol">
        <div className="methodology-card">
          <p className="methodology-text">
            All hypotheses were committed to Git <strong>before any training run</strong>. The git
            commit timestamp serves as tamper-proof evidence of honest methodology.
          </p>
          <div className="methodology-timeline">
            <div className="methodology-timeline__item">
              <span className="methodology-timeline__marker">1</span>
              <div>
                <strong>Write Hypotheses</strong>
                <p>Pre-register H1, H2, H3 with falsification criteria</p>
              </div>
            </div>
            <div className="methodology-timeline__item">
              <span className="methodology-timeline__marker">2</span>
              <div>
                <strong>Lock HYPOTHESES.md</strong>
                <p>Commit before any training — no post-hoc modifications</p>
              </div>
            </div>
            <div className="methodology-timeline__item">
              <span className="methodology-timeline__marker">3</span>
              <div>
                <strong>Train Models</strong>
                <p>R1 Baseline and R2 Gated, identical hyperparameters</p>
              </div>
            </div>
            <div className="methodology-timeline__item">
              <span className="methodology-timeline__marker">4</span>
              <div>
                <strong>Evaluate Once</strong>
                <p>Test set used exactly once per final model</p>
              </div>
            </div>
            <div className="methodology-timeline__item">
              <span className="methodology-timeline__marker">5</span>
              <div>
                <strong>Report Everything</strong>
                <p>Including honest negative results (H2)</p>
              </div>
            </div>
          </div>
        </div>
      </Section>

      <Section icon={Database} title="Data Pipeline">
        <div className="methodology-card">
          <h4>Zero Data Leakage Split</h4>
          <p className="methodology-text">
            DIV2K Valid HR images are split <strong>at the image level</strong> before any crop
            extraction, ensuring no pixel content leaks between splits.
          </p>
          <div className="methodology-splits">
            <div className="methodology-split">
              <span className="methodology-split__count">300</span>
              <span className="methodology-split__label">Training Crops</span>
              <span className="methodology-split__size">320×320 px</span>
            </div>
            <div className="methodology-split">
              <span className="methodology-split__count">20</span>
              <span className="methodology-split__label">Validation Crops</span>
              <span className="methodology-split__size">256×256 px</span>
            </div>
            <div className="methodology-split">
              <span className="methodology-split__count">50</span>
              <span className="methodology-split__label">Test Crops</span>
              <span className="methodology-split__size">256×256 px (locked)</span>
            </div>
          </div>
        </div>
      </Section>

      <Section icon={Layers} title="Corruption Model">
        <div className="methodology-card">
          <h4>Exact Integer Wave-Vector Sinusoidal Noise</h4>
          <p className="methodology-text">
            Corruption is a sum of sinusoidal patterns with controllable frequency (k), 
            orientation (θ), amplitude (A), and random phase (φ):
          </p>
          <div className="methodology-formula">
            <code>noise(x, y) = A · sin(2π · k · (x·cos(θ) + y·sin(θ)) / N + φ)</code>
          </div>
          <div className="methodology-params">
            <div className="methodology-param">
              <span className="methodology-param__name">Frequency (k)</span>
              <span className="methodology-param__train">Train: [8, 38]</span>
              <span className="methodology-param__test">Test: [3, 120]</span>
            </div>
            <div className="methodology-param">
              <span className="methodology-param__name">Amplitude (A)</span>
              <span className="methodology-param__train">Train: [0.05, 0.25]</span>
              <span className="methodology-param__test">Test: [0.05, 0.30]</span>
            </div>
            <div className="methodology-param">
              <span className="methodology-param__name">Orientation (θ)</span>
              <span className="methodology-param__train">Train: [0, π)</span>
              <span className="methodology-param__test">Test: [0, π) @ 7 steps</span>
            </div>
          </div>
        </div>
      </Section>

      <Section icon={TestTube} title="Models Compared">
        <div className="methodology-models-grid">
          <div className="methodology-model-card methodology-model-card--neural">
            <h4>R1 Baseline</h4>
            <span className="type-badge type-badge--neural">Neural</span>
            <p>Standard SFHformer-M architecture, trained 20K iterations with AdamW + Cosine Annealing</p>
            <ul>
              <li>No architectural modifications</li>
              <li>Learning rate: 1e-3 → 1e-6</li>
              <li>Batch size: 4</li>
              <li>Seed: 0</li>
            </ul>
          </div>
          <div className="methodology-model-card methodology-model-card--neural">
            <h4>R2 Gated</h4>
            <span className="type-badge type-badge--neural">Neural</span>
            <p>SFHformer-M + RadialBandGate inserted at FourierUnit's frequency branch</p>
            <ul>
              <li>Learnable log-amplitude radial mask</li>
              <li>Initialized to identity (all-ones)</li>
              <li>Same hyperparameters as R1</li>
            </ul>
          </div>
          <div className="methodology-model-card methodology-model-card--classical">
            <h4>Classical Baselines</h4>
            <span className="type-badge type-badge--classical">Classical</span>
            <p>Signal-processing approaches with no learning</p>
            <ul>
              <li>Blind FFT Notch Filter</li>
              <li>Gaussian Blur (σ=1.0, σ=2.0)</li>
              <li>Median Filter (3×3)</li>
              <li>Identity (do-nothing)</li>
            </ul>
          </div>
        </div>
      </Section>

      <Section icon={Shield} title="Falsification Criteria">
        <div className="methodology-card">
          <table className="data-table">
            <thead>
              <tr>
                <th>Hypothesis</th>
                <th>Would Be Falsified If…</th>
                <th>Actual Outcome</th>
              </tr>
            </thead>
            <tbody>
              {HYPOTHESES.map((h) => (
                <tr key={h.id}>
                  <td><strong>{h.id}</strong>: {h.title}</td>
                  <td className="methodology-falsify">{
                    h.id === 'H1' ? 'Equal or better ΔPSNR on out-of-band frequencies' :
                    h.id === 'H2' ? 'Improvement within CI width or in-band degrades' :
                    'Both neural methods beat notch by a large margin'
                  }</td>
                  <td>
                    <span className={`hypothesis-card__badge hypothesis-card__badge--${h.verdict === 'CONFIRMED' ? 'confirmed' : 'rejected'}`}>
                      {h.icon} {h.verdict}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
    </div>
  );
}
