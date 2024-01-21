/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.tensorflow.lite.examples.objectdetection

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.detector.Detection
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import com.googlecode.tesseract.android.TessBaseAPI
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream


class ObjectDetectorHelper(
    var threshold: Float = 0.5f,
    var numThreads: Int = 2,
    var maxResults: Int = 3,
    var currentDelegate: Int = 0,
    var currentModel: Int = 0,
    val context: Context,
    val objectDetectorListener: DetectorListener?
) {

    // For this example this needs to be a var so it can be reset on changes. If the ObjectDetector
    // will not change, a lazy val would be preferable.
    private var objectDetector: ObjectDetector? = null
    private val tessBaseAPI: TessBaseAPI = TessBaseAPI()

    init {
        setupObjectDetector()

        val dir = "${context.filesDir}/tesseract"
        if (checkLanguageFile("$dir/tessdata")) {
            val initSuccess = tessBaseAPI.init(dir, "slv")
            Log.d("Tesseract", "Tesseract initialization success: $initSuccess")
        }
    }

    private fun checkLanguageFile(dir: String): Boolean {
        val file = File(dir)
        if (!file.exists() && file.mkdirs()) createFiles(dir) else if (file.exists()) {
            val filePath = "$dir/slv.traineddata"
            val langDataFile = File(filePath)
            if (!langDataFile.exists()) createFiles(dir)
        }
        return true
    }

    private fun createFiles(dir: String) {
        val assetMgr = context.assets
        var inputStream: InputStream? = null
        var outputStream: OutputStream? = null
        try {
            inputStream = assetMgr.open("slv.traineddata")
            val destFile = "$dir/slv.traineddata"
            outputStream = FileOutputStream(destFile)
            val buffer = ByteArray(1024)
            var read: Int
            while (inputStream.read(buffer).also { read = it } != -1) {
                outputStream.write(buffer, 0, read)
            }
            inputStream.close()
            outputStream.flush()
            outputStream.close()
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    fun clearObjectDetector() {
        objectDetector = null
    }

    // Initialize the object detector using current settings on the
    // thread that is using it. CPU and NNAPI delegates can be used with detectors
    // that are created on the main thread and used on a background thread, but
    // the GPU delegate needs to be used on the thread that initialized the detector
    fun setupObjectDetector() {
        // Create the base options for the detector using specifies max results and score threshold
        val optionsBuilder =
            ObjectDetector.ObjectDetectorOptions.builder()
                .setScoreThreshold(threshold)
                .setMaxResults(maxResults)

        // Set general detection options, including number of used threads
        val baseOptionsBuilder = BaseOptions.builder().setNumThreads(numThreads)

        // Use the specified hardware for running the model. Default to CPU
        when (currentDelegate) {
            DELEGATE_CPU -> {
                // Default
            }
            DELEGATE_GPU -> {
                if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                    baseOptionsBuilder.useGpu()
                } else {
                    objectDetectorListener?.onError("GPU is not supported on this device")
                }
            }
            DELEGATE_NNAPI -> {
                baseOptionsBuilder.useNnapi()
            }
        }

        optionsBuilder.setBaseOptions(baseOptionsBuilder.build())

        val modelName =
            when (currentModel) {
//                MODEL_MOBILENETV1 -> "mobilenetv1.tflite"
//                MODEL_EFFICIENTDETV0 -> "efficientdet-lite0.tflite"
//                MODEL_EFFICIENTDETV1 -> "efficientdet-lite1.tflite"
//                MODEL_EFFICIENTDETV2 -> "fruit_v1.tflite"
//                MODEL_FRUIT_DETECTION -> "fruit_v2.tflite"
                Version1 -> "fruit_v1.tflite"
                Version2 -> "fruit_v2.tflite"
                else -> "fruit_v1.tflite"
            }

        try {
            objectDetector =
                ObjectDetector.createFromFileAndOptions(context, modelName, optionsBuilder.build())
        } catch (e: IllegalStateException) {
            objectDetectorListener?.onError(
                "Object detector failed to initialize. See error logs for details"
            )
            Log.e("Test", "TFLite failed to load model with error: " + e.message)
        }
    }



    fun detect(image: Bitmap, imageRotation: Int) {
        if (objectDetector == null) {
            setupObjectDetector()
        }

        // Inference time is the difference between the system time at the start and finish of the
        // process
        var inferenceTime = SystemClock.uptimeMillis()

        // Create preprocessor for the image.
        // See https://www.tensorflow.org/lite/inference_with_metadata/
        //            lite_support#imageprocessor_architecture
        val imageProcessor =
            ImageProcessor.Builder()
                .add(Rot90Op(-imageRotation / 90))
                .build()

        // Preprocess the image and convert it into a TensorImage for detection.
        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))

        val results = objectDetector?.detect(tensorImage)
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime


        val recognizedText = recognizeText(image)
        Log.d("Tesseract", "Recognized text: $recognizedText")

        objectDetectorListener?.onResults(
            results,
            inferenceTime,
            tensorImage.height,
            tensorImage.width,
            recognizedText
        )
    }

    fun recognizeText(bitmap: Bitmap): String {
        tessBaseAPI.setImage(bitmap)
        return tessBaseAPI.utF8Text
    }

    interface DetectorListener {
        fun onError(error: String)
        fun onResults(
            results: MutableList<Detection>?,
            inferenceTime: Long,
            imageHeight: Int,
            imageWidth: Int,
            recognizedText: String
        )
    }



    companion object {
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DELEGATE_NNAPI = 2
        //        const val MODEL_MOBILENETV1 = 0
//        const val MODEL_EFFICIENTDETV0 = 1
//        const val MODEL_EFFICIENTDETV1 = 2
//        const val MODEL_EFFICIENTDETV2 = 3
        const val Version1 = 0
        const val Version2 = 1
    }
}