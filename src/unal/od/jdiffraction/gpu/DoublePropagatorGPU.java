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
package unal.od.jdiffraction.gpu;

import jcuda.driver.CUdeviceptr;

/**
 * Abstract class for GPU diffraction calculation with double precision.
 *
 * @author Pablo Piedrahita-Quintero (jppiedrahitaq@unal.edu.co)
 * @author Carlos Trujillo (catrujila@unal.edu.co)
 * @author Jorge Garcia-Sucerquia (jigarcia@unal.edu.co)
 *
 * @since JDiffraction 1.2
 */
public abstract class DoublePropagatorGPU {

    /**
     * Performs numerical diffraction of the complex data in <code>field</code>,
     * leaving the result in <code>field</code>. The physical layout of the
     * complex data must be the same as in JTransforms:
     * <p>
     * {@code
     * field[i * 2 * N + 2 * j] = Re[i][j],
     * field[i * 2 * N + 2 * j + 1] = Im[i][j]; 0 &lt;= i &lt; M, 0 &lt;= j &lt; N
     * }
     *
     * @param field The complex field to diffract.
     */
    public abstract void diffract(double[] field);

    /**
     * Performs numerical diffraction of the complex data in
     * <code>devField</code>, leaving the result in <code>devField</code>.
     * <code>devField</code> must be a pointer to the data on the GPU memory.
     * The physical layout of the complex data must be the same as in
     * JTransforms:
     * <p>
     * {@code
     * field[i * 2 * N + 2 * j] = Re[i][j],
     * field[i * 2 * N + 2 * j + 1] = Im[i][j]; 0 &lt;= i &lt; M, 0 &lt;= j &lt; N
     * }
     *
     * @param devField The complex field to diffract.
     */
    public abstract void diffract(CUdeviceptr devField);
}
