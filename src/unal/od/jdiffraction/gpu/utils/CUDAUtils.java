/*
 * Copyright 2016 Universidad Nacional de Colombia
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package unal.od.jdiffraction.gpu.utils;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK;
import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuDeviceGetAttribute;
import static jcuda.driver.JCudaDriver.cuInit;

/**
 * JCuda Utilities class. Some functions were taken from JCuda examples.
 *
 * @author Pablo Piedrahita-Quintero (jppiedrahitaq@unal.edu.co)
 * @author Carlos Trujillo (catrujila@unal.edu.co)
 * @author Jorge Garcia-Sucerquia (jigarcia@unal.edu.co)
 *
 * @since JDiffraction 1.2
 */
public class CUDAUtils {

    private static CUDAUtils INSTANCE = null;

    private static boolean init = false;
    private static CUdevice device = null;
    private static CUcontext context = null;

    private CUDAUtils() {
    }

    private synchronized static void createInstance() {
        if (INSTANCE == null) {
            INSTANCE = new CUDAUtils();
        }
    }

    /**
     * Returns the current instance of CUDAUtils. If the current instance is
     * <code>null</code>, a new instance is created.
     *
     * @return instance of CUDAUtils
     */
    public static CUDAUtils getInstance() {
        if (INSTANCE == null) {
            createInstance();
        }
        return INSTANCE;
    }

    /**
     * Initialize the CUDA driver API.
     */
    public void initCUDA() {
        if (init) {
            return;
        }

        cuInit(0);

        init = true;
    }

    /**
     * Returns the <code>device</code> assigned to <code>ordinal</code>.
     *
     * @param ordinal
     * @return device
     */
    public CUdevice getDevice(int ordinal) {
//        if (device == null) {
        device = new CUdevice();
        cuDeviceGet(device, ordinal);
//        }

        return device;
    }

    /**
     * Returns the context for <code>device</code>. If the current context is
     * <code>null</code>, a new CUcontext is created.
     *
     * @param device
     * @return context
     */
    public CUcontext getContext(CUdevice device) {
        if (context == null) {
            context = new CUcontext();
            cuCtxCreate(context, 0, device);
        }

        return context;
    }

    /**
     * Returns the maximum number of threads per block for <code>device</code>.
     *
     * @param device
     * @return maximum number of threads
     */
    public int getMaxThreads(CUdevice device) {
        int[] maxThreadsArray = {0};
        cuDeviceGetAttribute(maxThreadsArray, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
        int maxThreads = maxThreadsArray[0];

        return maxThreads;
    }

    /**
     * The extension of the given file name is replaced with "ptx". If the file
     * with the resulting name does not exist, it is compiled from the given
     * file using NVCC. The name of the PTX file is returned.
     *
     * @param cuFileName The name of the .CU file
     * @return The name of the PTX file
     * @throws IOException If an I/O error occurs
     */
    public static String preparePtxFile(String cuFileName) throws IOException {
        int endIndex = cuFileName.lastIndexOf('.');
        if (endIndex == -1) {
            endIndex = cuFileName.length() - 1;
        }
        String ptxFileName = cuFileName.substring(0, endIndex + 1) + "ptx";
        File ptxFile = new File(ptxFileName);
        if (ptxFile.exists()) {
            return ptxFileName;
        }

        File cuFile = new File(cuFileName);
        if (!cuFile.exists()) {
            throw new IOException("Input file not found: " + cuFileName);
        }
        String modelString = "-m" + System.getProperty("sun.arch.data.model");
        String command
                = "nvcc " + modelString + " -ptx "
                + cuFile.getPath() + " -o " + ptxFileName;

        System.out.println("Executing\n" + command);
        Process process = Runtime.getRuntime().exec(command);

        String errorMessage
                = new String(toByteArray(process.getErrorStream()));
        String outputMessage
                = new String(toByteArray(process.getInputStream()));
        int exitValue = 0;
        try {
            exitValue = process.waitFor();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException(
                    "Interrupted while waiting for nvcc output", e);
        }

        if (exitValue != 0) {
            System.out.println("nvcc process exitValue " + exitValue);
            System.out.println("errorMessage:\n" + errorMessage);
            System.out.println("outputMessage:\n" + outputMessage);
            throw new IOException(
                    "Could not create .ptx file: " + errorMessage);
        }

        System.out.println("Finished creating PTX file");
        return ptxFileName;
    }

    /**
     * Fully reads the given InputStream and returns it as a byte array.
     *
     * @param inputStream The input stream to read
     * @return The byte array containing the data from the input stream
     * @throws IOException If an I/O error occurs
     */
    private static byte[] toByteArray(InputStream inputStream) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte buffer[] = new byte[8192];
        while (true) {
            int read = inputStream.read(buffer);
            if (read == -1) {
                break;
            }
            baos.write(buffer, 0, read);
        }
        return baos.toByteArray();
    }
}
