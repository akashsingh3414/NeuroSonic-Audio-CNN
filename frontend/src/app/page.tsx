"use client"

import type React from "react"

import { useState } from "react"
import ColorScale from "~/components/ColorScale"
import FeatureMap from "~/components/FeatureMap"
import { Badge } from "~/components/ui/badge"
import { Button } from "~/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card"
import { Progress } from "~/components/ui/progress"
import Waveform from "~/components/Waveform"

interface Prediction {
  class: string
  confidence: number
}

interface LayerData {
  shape: number[]
  values: number[][]
}

type VisualizationData = Record<string, LayerData>

interface WaveformData {
  values: number[]
  sample_rate: number
  duration: number
}

interface ApiResponse {
  predictions: Prediction[]
  visualization: VisualizationData
  input_spectrogram: LayerData
  waveform: WaveformData
}

function splitLayers(visualization: VisualizationData) {
  const main: [string, LayerData][] = []
  const internals: Record<string, [string, LayerData][]> = {}

  for (const [name, data] of Object.entries(visualization)) {
    if (!name.includes(".")) {
      main.push([name, data])
    } else {
      const [parent] = name.split(".")
      if (parent === undefined) continue

      internals[parent] ??= []
      internals[parent].push([name, data])
    }
  }

  return { main, internals }
}

export default function HomePage() {
  const [vizData, setVizData] = useState<ApiResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [fileName, setFileName] = useState("")
  const [error, setError] = useState<string | null>(null)
  const [audioURL, setAudioURL] = useState<string | null>(null)

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    setFileName(file.name)
    setIsLoading(true)
    setError(null)
    setVizData(null)
    setAudioURL(null)

    const reader = new FileReader()
    reader.readAsArrayBuffer(file)
    reader.onload = async () => {
      try {
        const arrayBuffer = reader.result as ArrayBuffer
        const base64String = btoa(
          new Uint8Array(arrayBuffer).reduce((data, byte) => data + String.fromCharCode(byte), ""),
        )

        const BASE_URL = process.env.NEXT_PUBLIC_BASE_URL ?? "http://127.0.0.1:8000"

        const response = await fetch(`${BASE_URL}/inference`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ audio_data: base64String }),
        })

        if (!response.ok) {
          throw new Error(`API error ${response.statusText}`)
        }

        const data = (await response.json()) as ApiResponse
        setVizData(data)

        const url = URL.createObjectURL(file)
        setAudioURL(url)
      } catch (err) {
        setError(err instanceof Error ? err.message : "An unknown error occurred")
      } finally {
        setIsLoading(false)
      }
    }
    reader.onerror = () => {
      setError("Failed to read the file.")
      setIsLoading(false)
    }
  }

  const { main, internals } = vizData ? splitLayers(vizData.visualization) : { main: [], internals: {} }

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-6 md:p-8">
      <div className="mx-auto max-w-7xl">
        <div className="mb-12">
          <h1 className="mb-2 text-3xl font-bold tracking-tight text-slate-900 md:text-4xl">CNN Audio Visualizer</h1>
          <p className="text-base text-slate-600">
            Upload a WAV file to analyze audio with deep learning and visualize CNN feature maps
          </p>
        </div>

        <Card className="mb-8 border-slate-200 bg-white shadow-sm">
          <CardContent className="pt-8">
            <div className="flex flex-col items-center gap-6">
              <div className="relative inline-block w-full max-w-xs">
                <input
                  type="file"
                  accept=".wav"
                  id="file-upload"
                  onChange={handleFileChange}
                  disabled={isLoading}
                  className="absolute inset-0 w-full cursor-pointer opacity-0"
                />
                <Button disabled={isLoading} className="w-full bg-blue-600 hover:bg-blue-700" size="lg">
                  {isLoading ? (
                    <span className="flex items-center gap-2">
                      <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                      Analysing...
                    </span>
                  ) : (
                    "Choose WAV File"
                  )}
                </Button>
              </div>

              {fileName && <Badge className="bg-slate-200 text-slate-800">📁 {fileName}</Badge>}

              {audioURL && (
                <div className="w-full max-w-xs">
                  <audio controls src={audioURL} className="w-full rounded-lg bg-slate-100">
                    Your browser does not support the audio element.
                  </audio>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {error && (
          <Card className="mb-8 border-red-200 bg-red-50 shadow-sm">
            <CardContent className="pt-6">
              <p className="text-red-700">⚠️ Error: {error}</p>
            </CardContent>
          </Card>
        )}

        {vizData && (
          <div className="space-y-8">
            <Card className="border-slate-200 bg-white shadow-sm">
              <CardHeader>
                <CardTitle className="text-slate-900">Top Predictions</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {vizData.predictions.slice(0, 3).map((pred, i) => (
                    <div key={pred.class} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="text-sm font-semibold text-slate-700">{pred.class.replaceAll("_", " ")}</div>
                        <Badge className={i === 0 ? "bg-blue-600 text-white" : "bg-slate-200 text-slate-800"}>
                          {(pred.confidence * 100).toFixed(1)}%
                        </Badge>
                      </div>
                      <Progress value={pred.confidence * 100} className="h-2 bg-slate-200" />
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              <Card className="border-slate-200 bg-white shadow-sm">
                <CardHeader>
                  <CardTitle className="text-slate-900">Input Spectrogram</CardTitle>
                </CardHeader>
                <CardContent>
                  <FeatureMap
                    data={vizData.input_spectrogram.values}
                    title={`${vizData.input_spectrogram.shape.join(" × ")}`}
                    spectrogram
                  />
                  <div className="mt-6 flex justify-end">
                    <ColorScale width={200} height={16} min={-1} max={1} />
                  </div>
                </CardContent>
              </Card>
              <Card className="border-slate-200 bg-white shadow-sm">
                <CardHeader>
                  <CardTitle className="text-slate-900">Audio Waveform</CardTitle>
                </CardHeader>
                <CardContent>
                  <Waveform
                    data={vizData.waveform.values}
                    title={`${vizData.waveform.duration.toFixed(2)}s @ ${vizData.waveform.sample_rate}Hz`}
                  />
                </CardContent>
              </Card>
            </div>

            <Card className="border-slate-200 bg-white shadow-sm">
              <CardHeader>
                <CardTitle className="text-slate-900">Convolutional Layer Outputs</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-6 md:grid-cols-3 lg:grid-cols-5">
                  {main.map(([mainName, mainData]) => (
                    <div key={mainName} className="space-y-4">
                      <div>
                        <h4 className="mb-3 text-sm font-semibold text-slate-700">{mainName}</h4>
                        <FeatureMap data={mainData.values} title={`${mainData.shape.join(" × ")}`} />
                      </div>

                      {internals[mainName] && (
                        <div className="max-h-80 overflow-y-auto rounded-lg border border-slate-200 bg-slate-50 p-3">
                          <div className="space-y-2">
                            {internals[mainName]
                              .sort(([a], [b]) => a.localeCompare(b))
                              .map(([layerName, layerData]) => (
                                <FeatureMap
                                  key={layerName}
                                  data={layerData.values}
                                  title={layerName.replace(`${mainName}.`, "")}
                                  internal={true}
                                />
                              ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
                <div className="mt-6 flex justify-end">
                  <ColorScale width={200} height={16} min={-1} max={1} />
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </main>
  )
}
