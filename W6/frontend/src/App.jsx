import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity,
  ArrowRight,
  Brain,
  Download,
  Gauge,
  Github,
  Image as ImageIcon,
  RefreshCcw,
  SlidersHorizontal,
  Sparkles,
  UploadCloud,
  X,
} from 'lucide-react'
import { denoiseImage, getHealth, getModelInfo } from './services/api'

const noisePresets = [
  { label: 'Low', value: 0.15 },
  { label: 'Medium', value: 0.3 },
  { label: 'High', value: 0.45 },
]

const modelPipeline = [
  'Input Image',
  'Preprocess',
  '28 × 28 Grayscale',
  'Gaussian Noise',
  'Noisy Image',
  'Encoder',
  'Latent Space',
  'Decoder',
  'Denoised Image',
  'Quality Metrics',
]

function App() {
  const inputRef = useRef(null)
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [noiseFactor, setNoiseFactor] = useState(0.3)
  const [results, setResults] = useState(null)
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [comparisonMode, setComparisonMode] = useState('side-by-side')
  const [modelInfo, setModelInfo] = useState(null)
  const [health, setHealth] = useState(null)

  useEffect(() => {
    async function loadMeta() {
      try {
        const [healthResponse, modelResponse] = await Promise.all([getHealth(), getModelInfo()])
        setHealth(healthResponse.data)
        setModelInfo(modelResponse.data)
      } catch {
        setError('Unable to reach the backend model service.')
      }
    }

    loadMeta()
  }, [])

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl)
    }
  }, [previewUrl])

  const metrics = useMemo(() => {
    if (!results?.metrics) return []

    return [
      {
        label: 'Noisy MSE',
        value: `${results.metrics.noisy_mse.toFixed(4)}`,
        icon: Gauge,
        note: 'Difference between original and noisy image.',
      },
      {
        label: 'Denoised MSE',
        value: `${results.metrics.denoised_mse.toFixed(4)}`,
        icon: Activity,
        note: 'Difference between original and reconstructed image.',
      },
      {
        label: 'PSNR',
        value: `${results.metrics.psnr.toFixed(2)} dB`,
        icon: Sparkles,
        note: 'Higher values indicate better reconstruction quality.',
      },
      {
        label: 'Improvement',
        value: `${results.metrics.improvement_percentage.toFixed(2)}%`,
        icon: Brain,
        note: 'Measured reduction in error after denoising.',
      },
    ]
  }, [results])

  function validateFile(file) {
    if (!file) return 'Please upload an image first.'
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg']
    if (!allowedTypes.includes(file.type)) {
      return 'Unsupported file type. Please upload a PNG or JPG image.'
    }
    if (file.size > 5 * 1024 * 1024) {
      return 'File is too large. Please keep uploads under 5 MB.'
    }
    return ''
  }

  function handleFileChange(file) {
    const validationMessage = validateFile(file)
    if (validationMessage) {
      setError(validationMessage)
      return
    }

    if (previewUrl) URL.revokeObjectURL(previewUrl)
    const nextPreview = URL.createObjectURL(file)
    setPreviewUrl(nextPreview)
    setSelectedFile(file)
    setError('')
    setResults(null)
  }

  async function handleProcess() {
    const validationMessage = validateFile(selectedFile)
    if (validationMessage) {
      setError(validationMessage)
      return
    }

    if (!health?.model_loaded) {
      setError('The backend model is not yet available. Please start the API service.')
      return
    }

    setIsLoading(true)
    setError('')

    try {
      const response = await denoiseImage({ file: selectedFile, noiseFactor })
      setResults(response.data)
    } catch (apiError) {
      setError(apiError?.response?.data?.detail || 'The request failed at the backend. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  function handleReset() {
    setSelectedFile(null)
    setResults(null)
    setError('')
    setNoiseFactor(0.3)
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl('')
  }

  function handleDownload(filename, dataUrl) {
    const link = document.createElement('a')
    link.href = dataUrl
    link.download = filename
    link.click()
  }

  return (
    <div className="app-shell">
      <nav className="topbar">
        <div>
          <p className="eyebrow">AI-Powered Image Restoration</p>
          <h1>MNIST Denoise AI</h1>
        </div>
        <div className="nav-links">
          <a href="#home">Home</a>
          <a href="#how-it-works">How It Works</a>
          <a href="#model">Model</a>
          <a href="#about">About</a>
          <a className="github-btn" href="https://github.com" target="_blank" rel="noreferrer" aria-label="GitHub">
            <Github size={16} />
          </a>
        </div>
      </nav>

      <section className="hero-section" id="home">
        <div className="hero-copy">
          <p className="eyebrow">Deep Learning Image Denoising using Autoencoder</p>
          <h2>Remove Image Noise with Deep Learning</h2>
          <p className="hero-text">
            Upload a handwritten digit image, simulate noise, and watch a deep learning autoencoder reconstruct a cleaner version of the image.
          </p>
          <div className="cta-row">
            <a className="primary-btn" href="#studio">Try Denoising</a>
            <a className="ghost-btn" href="#how-it-works">How It Works</a>
          </div>
        </div>
        <div className="hero-visual">
          <div className="mini-card">
            <span>Original</span>
            <div className="mini-image gradient-1" />
          </div>
          <ArrowRight size={20} />
          <div className="mini-card">
            <span>Noisy</span>
            <div className="mini-image gradient-2" />
          </div>
          <ArrowRight size={20} />
          <div className="mini-card">
            <span>Denoised</span>
            <div className="mini-image gradient-3" />
          </div>
        </div>
      </section>

      <section className="studio-card" id="studio">
        <div className="section-header">
          <div>
            <p className="eyebrow">Image Denoising Studio</p>
            <h3>Upload an MNIST-style digit image and process it using the trained autoencoder.</h3>
          </div>
        </div>

        <div className="workspace-grid">
          <div className="controls-panel glass-card">
            <div className="upload-area" onClick={() => inputRef.current?.click()} onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); handleFileChange(e.dataTransfer.files?.[0]) }}>
              <UploadCloud size={36} />
              <strong>Upload an MNIST Digit</strong>
              <span>Drag &amp; drop an image here or click to browse</span>
              <small>PNG · JPG · JPEG · Max 5 MB</small>
              <input ref={inputRef} type="file" accept="image/png,image/jpeg,image/jpg" hidden onChange={(e) => handleFileChange(e.target.files?.[0])} />
            </div>

            {selectedFile && (
              <div className="file-meta">
                <div>
                  <strong>{selectedFile.name}</strong>
                  <span>{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</span>
                </div>
                <button type="button" className="remove-btn" onClick={handleReset}>
                  <X size={16} />
                  Remove
                </button>
              </div>
            )}

            {previewUrl && (
              <div className="preview-box">
                <img src={previewUrl} alt="Uploaded MNIST preview" />
              </div>
            )}

            <div className="config-block">
              <div className="label-row">
                <span>Noise Configuration</span>
                <span>{noiseFactor.toFixed(1)}</span>
              </div>
              <input type="range" min="0" max="0.6" step="0.05" value={noiseFactor} onChange={(e) => setNoiseFactor(Number(e.target.value))} />
              <div className="preset-row">
                {noisePresets.map((preset) => (
                  <button key={preset.label} type="button" className={noiseFactor === preset.value ? 'preset active' : 'preset'} onClick={() => setNoiseFactor(preset.value)}>
                    {preset.label}
                  </button>
                ))}
              </div>
              <p className="helper-text">Gaussian noise is added to simulate image corruption before the denoising process.</p>
            </div>

            <button type="button" className="process-btn" onClick={handleProcess} disabled={isLoading}>
              <Sparkles size={18} />
              {isLoading ? 'Running Autoencoder...' : 'Denoise Image'}
            </button>

            <button type="button" className="secondary-btn" onClick={handleReset}>
              <RefreshCcw size={16} />
              Reset
            </button>

            {error && <div className="error-box" role="alert">{error}</div>}
          </div>

          <div className="results-panel glass-card">
            <div className="comparison-mode-row">
              <button type="button" className={comparisonMode === 'side-by-side' ? 'mode-btn active' : 'mode-btn'} onClick={() => setComparisonMode('side-by-side')}>Side-by-Side</button>
              <button type="button" className={comparisonMode === 'slider' ? 'mode-btn active' : 'mode-btn'} onClick={() => setComparisonMode('slider')}>Comparison Slider</button>
            </div>

            <div className="results-grid">
              <ResultCard title="Original" subtitle="28 × 28 grayscale input" image={results?.original_image ? `data:image/png;base64,${results.original_image}` : previewUrl} />
              <ResultCard title="Noisy Input" subtitle={`Noise Factor: ${noiseFactor.toFixed(1)}`} image={results?.noisy_image ? `data:image/png;base64,${results.noisy_image}` : previewUrl} />
              <ResultCard title="Denoised Output" subtitle="Reconstructed by Autoencoder" image={results?.denoised_image ? `data:image/png;base64,${results.denoised_image}` : ''} />
            </div>

            {results?.comparison_image && comparisonMode === 'slider' ? (
              <div className="slider-wrapper">
                <div className="comparison-slider">
                  <img src={`data:image/png;base64,${results.comparison_image}`} alt="Comparison image" />
                </div>
              </div>
            ) : null}

            {results && (
              <div className="download-row">
                <button type="button" className="secondary-btn" onClick={() => handleDownload('denoised.png', `data:image/png;base64,${results.denoised_image}`)}>
                  <Download size={16} />
                  Download Denoised Image
                </button>
                <button type="button" className="secondary-btn" onClick={() => handleDownload('comparison.png', `data:image/png;base64,${results.comparison_image}`)}>
                  <Download size={16} />
                  Download Comparison
                </button>
              </div>
            )}
          </div>
        </div>
      </section>

      <section className="metrics-section">
        <div className="section-header">
          <div>
            <p className="eyebrow">Quality Metrics</p>
            <h3>Model performance on the current denoising task</h3>
          </div>
        </div>
        <div className="metrics-grid">
          {metrics.map((metric) => {
            const Icon = metric.icon
            return (
              <div className="metric-card glass-card" key={metric.label}>
                <Icon size={20} />
                <span>{metric.label}</span>
                <strong>{metric.value}</strong>
                <small>{metric.note}</small>
              </div>
            )
          })}
        </div>
      </section>

      <section className="info-grid" id="model">
        <div className="glass-card section-card">
          <p className="eyebrow">About the Autoencoder</p>
          <h3>Architecture Overview</h3>
          <div className="pipeline-box">
            {modelPipeline.map((item, index) => (
              <div key={item} className="pipeline-step">
                <span>{item}</span>
                {index < modelPipeline.length - 1 && <ArrowRight size={16} />}
              </div>
            ))}
          </div>
          <div className="model-meta-grid">
            <div><strong>Architecture:</strong> {modelInfo?.architecture || 'Fully Connected Autoencoder'}</div>
            <div><strong>Input:</strong> {modelInfo?.input || '28 × 28 grayscale image'}</div>
            <div><strong>Latent Dimension:</strong> {modelInfo?.latent_dimension || 32}</div>
            <div><strong>Activation:</strong> {modelInfo?.activation || 'ReLU + Sigmoid'}</div>
            <div><strong>Optimizer:</strong> Adam</div>
            <div><strong>Loss Function:</strong> {modelInfo?.loss_function || 'Mean Squared Error'}</div>
            <div><strong>Dataset:</strong> {modelInfo?.dataset || 'MNIST'}</div>
            <div><strong>Task:</strong> {modelInfo?.task || 'Image Denoising'}</div>
          </div>
        </div>

        <div className="glass-card section-card" id="how-it-works">
          <p className="eyebrow">How It Works</p>
          <h3>Four-step denoising flow</h3>
          <div className="flow-list">
            <div><strong>1 — Upload</strong><span>User uploads an MNIST-style handwritten digit image.</span></div>
            <div><strong>2 — Add Noise</strong><span>Gaussian noise is added to simulate a corrupted image.</span></div>
            <div><strong>3 — Encode and Decode</strong><span>The autoencoder compresses the noisy image into a latent representation and reconstructs it.</span></div>
            <div><strong>4 — Denoise</strong><span>The reconstructed output is compared with the original image to evaluate the denoising performance.</span></div>
          </div>
        </div>
      </section>

      <section className="about-section" id="about">
        <div className="glass-card section-card">
          <p className="eyebrow">About Project</p>
          <h3>MNIST Denoise AI</h3>
          <p>
            MNIST Denoise AI is a deep learning-based image restoration system that demonstrates how autoencoders can learn to reconstruct clean images from noisy inputs. The system is trained using the MNIST handwritten digit dataset, where noisy images are used as input and original clean images are used as target outputs.
          </p>
          <div className="tech-pills">
            <span>Python</span>
            <span>TensorFlow</span>
            <span>Keras</span>
            <span>React</span>
            <span>FastAPI</span>
            <span>NumPy</span>
            <span>Pillow</span>
          </div>
        </div>
      </section>
    </div>
  )
}

function ResultCard({ title, subtitle, image }) {
  return (
    <div className="result-card">
      <div className="result-heading">
        <strong>{title}</strong>
        <span>{subtitle}</span>
      </div>
      <div className="image-frame">
        {image ? <img src={image} alt={title} /> : <ImageIcon size={40} />}
      </div>
    </div>
  )
}

export default App
