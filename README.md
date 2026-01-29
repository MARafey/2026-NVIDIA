# NVIDIA iQuHACK 2026 Challenge

## Overview

The Low Autocorrelation of Binary Sequences (LABS) problem is a notoriously difficult optimization challenge, critical for high-performance radar and telecommunications.

Your objective is to take the current classical state-of-the-art Memetic Tabu Search (MTS) and evolve it. Rather than jumping to a purely quantum solution, you will engineer a hybrid quantum-enhanced workflow where samples from a quantum algorithm are used to seed the classical MTS population. You must then push the limits of performance by GPU-accelerating both the quantum simulation and the classical search components.

**We want you to vibe code!**
In modern R&D and this challenge, speed matters, but rigor and coordination matter more. We expect you to employ Agentic Strategies, utilizing AI tools that can reason across your codebase to act as your collaborators while you operate as the Technical Leadership Team. Your collective job is to decompose the problem, delegate tasks across your team and AI agents, and most importantly verify the work. As Leads, you must clearly communicate your planning, workflow, and solution, ensuring your team remains aligned and ready to pivot even as technical challenges shift your strategy.

## Logistics, Milestones, and Evaluation

In this challenge, you will mimic a real-world R&D pipeline, moving from rapid prototyping to high-performance deployment. You will utilize two distinct platforms, each chosen for a specific phase of your development lifecycle. 

* **Phase 1 (Prototyping): qBraid**

    For the "Ramp Up" and initial CPU validation, you will work on Milestones 1 and 2 in [qBraid](https://account-v2.qbraid.com/). qBraid is your "Dev Environment" — a zero-setup, pre-configured cudaq sandbox that allows you to focus entirely on mastering the algorithm and logic without worrying about infrastructure overhead.

    ### Milestones:
    1. **Ramp Up**: Master the state-of-the-art for LABS via a scaffolded tutorial.
    
    2. **Research & Plan**: Perform due diligence to design a custom quantum strategy and acceleration plan.

* **Phase 2 (Acceleration): Brev**

    Once your logic is validated, you will "graduate" your code to [Brev](https://brev.nvidia.com/) to complete Milestone 3 and 4. Brev provides on-demand access to a wide variety of NVIDIA GPU architectures (L4s, T4s, A100s, ...). We have provided a pre-configured GPU environment for this environment called a **Launchable**. You will use Brev to test your solution across different hardware configurations and unlock full GPU acceleration.

    ### Milestones: 
    3. **Build**: Validate your algorithm on a CPU in qBraid in the previous milestone, then migrate to Brev to deploy full GPU acceleration.

    4. **Showcase and Retrospective**: Present your solution, performance metrics, and your AI-driven workflow.

**Good luck. Let the agents build the code, you build the architecture.**

## Accessing Phase 1 of the Challenge with qBraid

<a href="https://account-v2.qbraid.com/?gitHubUrl=https://github.com/iQuHACK/2026-NVIDIA" target="_parent"><img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" alt="Launch On qBraid" width="150"/></a>

During the duration of the hackathon, you will have access to the new and improved version of the qBraid platform accessible through here: https://account-v2.qbraid.com/ 

If any issues occur, try deleting cache in your browser and refreshing the page.

### Steps for qBraid Environment Setup:

1. Click the `Launch on qBraid` button above <img align="right" width= "33%" src="images/image.png">

2. Navigate to GIT in the left sidebar and clone this repository

3. Add the CUDA-Q environment by navigating to the ENVS tab in the right sidebar, and click on `+ ADD`

4. Navigate to `CUDA-Q and GPU Quantum Environments`

5. Install `CUDA-Q (v0.13.0)` - this will take a few minutes

6. Navigate to `CUDA-Q and GPU Quantum Environments` <img align="right" width ="33%" src="images/image-2.png">

7. Once installation is complete, open the `labs_tutorial.ipynb` notebook

8. In the bottom left corner, make sure the kernel is set to `Python 3 [cuda q-v0.13.0]` 

9. Happy Hacking!


For any questions or additional assistance using qBraid, see the [lab documentation](https://docs.qbraid.com/lab/user-guide/getting-started), or reach out to the qBraid team on discord. 


## Accessing Phase 2 of the Challenge with Brev

<a href="https://brev.nvidia.com/" target="_parent"><img src="https://brev-assets.s3.us-west-1.amazonaws.com/nv-lb-dark.svg" width="150"/></a>

Congratulations on finishing the first part of the challenge! 

You will now get to run your code on real GPUs using NVIDIA's Brev Platform. Don't worry, you don't need to pay for anything! Once completing Phase 1 and your logic is validated, we will provide you with a **$20 Brev coupon code**.

### Steps **ONLY** to be completed by the **GPU Acceleration PIC**:
The designated GPU Acceleration PIC will create an organization in Brev, redeem the $20 coupon code, and provide their team members with access. 

1. Go to http://brev.nvidia.com and input your email to create an account 

<img align="right" width ="10%" src="images/image-3.png">

2. Create a new Brev organization by clicking on the building icon in the top right corner and selecting `+ Create a new organization.` 

3. Name your organization in the format: `MIT-<team_name_here>` Example: `MIT-qrazy-qubits` <img align="right" width ="39%" src="images/image-5.png">

4. Go to the `Team` tab located at the top. 

5. Click `Generate Invite Link` and share with your team members.

6. Go to the `Billing` tab 

7. Scroll down and click `Redeem Code` 

8. Enter the code you were provided in the `Enter Code` field. Please ensure coupon code is all lower case

<img style="float: right;" width ="29%" src="images/image-6.png">

9. Click `Redeem`

10. Click the `Deploy Now` Button above to access the materials in this repository in a pre-configured GPU-environment.

    * Before deploying the launchable, select `View All Options` to change your GPU selection.

    * ***Take budget into consideration when selecting a GPU to run your code on. We know it's tempting to select a B300, but selecting more expensive options will burn through your credits significantly faster.***

11. After selecting GPU configuration, click `Deploy Launchable`. <img style="float: right;" width ="30%" src="images/image-7.png">

    * You can check the status of your deployment by clicking `Go to Instance Page` or through the `GPUs` tab 

<img style="float: right;" width ="50%" src="images/image-8.png">

12. Once deployment is complete, you will see a GPU environment under the  `GPUs` tab.
    * For the GPU Acceleration PIC, this will show up under the `Mine` tab.

    * For all other team members, this will show up under the `Team` tab.

13. Select `Access Notebook` to pull up the notebook environment similar to qBraid. Team mebers should also complete this step. 

    <img style="float: right;" width ="55%" src="images/image-9.png">

    * If team members run into an error when trying to access the environment, go to the instance page, and scroll down to `Using Secure Links`.

    * Click `Edit Access` and enter your team member's email or username. You may need to try both.

14. Once all team members have access to the notebook, you can all edit the same notebook! To see your teammates' changes, refresh the notebook.

15. Happy Hacking!

For any questions or additional assistance using Brev, see the [Brev documentation](https://docs.nvidia.com/brev/latest/) or reach out to the NVIDIA team on discord.

## Resources
### CUDA-Q    

* If you are new to quantum computing, (e.g., you'd like a review of the definition of a qubit, quantum gates, and quantum circuits), then run through the first two notebooks of the [Quick Start to Quantum Computing](https://github.com/NVIDIA/cuda-q-academic/tree/main/quick-start-to-quantum) series.

* If you have quantum computing background and want a quick visual guide for translating a quantum circuit into a CUDA-Q kernel, check out this [hello world visualization tool](https://nvidia.github.io/cuda-q-academic/quick-start-to-quantum/interactive_widget/cudaq-hello-world.html).  For more in depth coverage of cuda-q syntax and examples, we recommend notebook 1 of the [QAOA for Max Cut series](https://github.com/NVIDIA/cuda-q-academic/tree/main/qaoa-for-max-cut) series as well as the [examples](https://nvidia.github.io/cuda-quantum/latest/using/examples/examples.html) and [applications](https://nvidia.github.io/cuda-quantum/latest/using/applications.html) in the CUDA-Q documentation, in particular the [QAOA example](https://nvidia.github.io/cuda-quantum/latest/applications/python/qaoa.html).  

* If you prefer to learn by watching, you can check out [minutes 30:40-38:28 of this demo](https://www.nvidia.com/en-us/on-demand/session/gtcdc25-dct51159/?playlistId=gtcdc25-quantum-computing-and-hpc&start=1840&end=2308) of description of cudaq kernels, sampling, and getting the state vector or watch [minutes 9:20-19:00 of this demo](https://www.youtube.com/live/DqPC-nlcXKA?si=ualhUnFYjW9BlbQz&t=560).


## Accessing Material Post Challenge

Challenge materials can be accessed via https://account-v2.qbraid.com/ or through https://account.qbraid.com/ once v2 is merged into the main platform.

If you completed Phase 2 of the challenge, materials can be accessed via https://brev.nvidia.com/.
