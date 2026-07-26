import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export async function getHealth() {
  return api.get('/api/health')
}

export async function getModelInfo() {
  return api.get('/api/model-info')
}

export async function denoiseImage({ file, noiseFactor }) {
  const formData = new FormData()
  formData.append('image', file)
  formData.append('noise_factor', Number(noiseFactor).toString())

  return api.post('/api/denoise', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
}

export default api
