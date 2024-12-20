import logging
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, Optional, Union

import typeguard

from parsl.jobs.states import JobState, JobStatus
from parsl.launchers import SingleNodeLauncher
from parsl.launchers.base import Launcher
from parsl.providers.cluster_provider import ClusterProvider
from parsl.providers.errors import SubmitException
from parsl.providers.slurm.template import template_string
from parsl.utils import RepresentationMixin, wtime_to_minutes


from sfapi_client import Client
from sfapi_client.compute import Machine
from sfapi_client.paths import RemotePath
from sfapi_client.jobs import JobCommand, JobState as NerscJobState


logger = logging.getLogger(__name__)

# Translate from the NERSC (slurm sacct) job status to Parsl job status
translate_table = {
    NerscJobState.PENDING: JobState.PENDING,
    NerscJobState.RUNNING: JobState.RUNNING,
    NerscJobState.CANCELLED: JobState.CANCELLED,
    NerscJobState.COMPLETED: JobState.COMPLETED,
    NerscJobState.FAILED: JobState.FAILED,
    NerscJobState.NODE_FAIL: JobState.FAILED,
    NerscJobState.BOOT_FAIL: JobState.FAILED,
    NerscJobState.DEADLINE: JobState.TIMEOUT,
    NerscJobState.TIMEOUT: JobState.TIMEOUT,
    NerscJobState.REVOKED: JobState.FAILED,
    NerscJobState.OUT_OF_MEMORY: JobState.FAILED,
    NerscJobState.SUSPENDED: JobState.HELD,
    NerscJobState.PREEMPTED: JobState.TIMEOUT,
    NerscJobState.REQUEUED: JobState.PENDING
}


class NerscSfapiProvider(ClusterProvider, RepresentationMixin):
    """NERSC SFAPI Execution Provider

    This provider uses the NERSC superfacility api (SFAPI) to submit, 
    check status and cancel jobs. The sbatch script to be used 
    is created from a template file in this same module.

    Parameters
    ----------
    partition : str
        Slurm partition to request blocks from. If unspecified or ``None``, no partition slurm directive will be specified.
    account : str
        Slurm account to which to charge resources used by the job. If unspecified or ``None``, the job will use the
        user's default account.
    qos : str
        Slurm queue to place job in. If unspecified or ``None``, no queue slurm directive will be specified.
    constraint : str
        Slurm job constraint, often used to choose cpu or gpu type. If unspecified or ``None``, no constraint slurm directive will be added.
    clusters : str
        Slurm cluster name, or comma seperated cluster list, used to choose between different clusters in a federated Slurm instance.
        If unspecified or ``None``, no slurm directive for clusters will be added.
    nodes_per_block : int
        Nodes to provision per block.
    cores_per_node : int
        Specify the number of cores to provision per node. If set to None, executors
        will assume all cores on the node are available for computation. Default is None.
    mem_per_node : int
        Specify the real memory to provision per node in GB. If set to None, no
        explicit request to the scheduler will be made. Default is None.
    init_blocks : int
        Number of blocks to provision at the start of the run. Default is 1.
    min_blocks : int
        Minimum number of blocks to maintain.
    max_blocks : int
        Maximum number of blocks to maintain.
    parallelism : float
        Ratio of provisioned task slots to active tasks. A parallelism value of 1 represents aggressive
        scaling where as many resources as possible are used; parallelism close to 0 represents
        the opposite situation in which as few resources as possible (i.e., min_blocks) are used.
    walltime : str
        Walltime requested per block in HH:MM:SS.
    scheduler_options : str
        String to prepend to the #SBATCH blocks in the submit script to the scheduler.
    worker_init : str
        Command to be run before starting a worker, such as 'module load Anaconda; source activate env'.
    exclusive : bool (Default = True)
        Requests nodes which are not shared with other running jobs.
    launcher : Launcher
        Launcher for this provider. Possible launchers include
        :class:`~parsl.launchers.SingleNodeLauncher` (the default),
        :class:`~parsl.launchers.SrunLauncher`, or
        :class:`~parsl.launchers.AprunLauncher`
    """

    @typeguard.typechecked
    def __init__(self,
                 key: Union[str, Path],
                 machine: Optional[Union[str, Machine]] = Machine.perlmutter,
                 workdir: Optional[Union[Path, RemotePath, str]] = None,
                 account: Optional[str] = None,
                 qos: Optional[str] = None,
                 constraint: Optional[str] = None,
                 nodes_per_block: int = 1,
                 cores_per_node: Optional[int] = None,
                 mem_per_node: Optional[int] = None,
                 init_blocks: int = 1,
                 min_blocks: int = 0,
                 max_blocks: int = 1,
                 parallelism: float = 1,
                 walltime: str = "00:10:00",
                 scheduler_options: str = '',
                 worker_init: str = '',
                 cmd_timeout: int = 10,
                 exclusive: bool = True,
                 launcher: Launcher = SingleNodeLauncher()):
        label = 'nersc_sfapi'
        super().__init__(label,
                         nodes_per_block,
                         init_blocks,
                         min_blocks,
                         max_blocks,
                         parallelism,
                         walltime,
                         cmd_timeout=cmd_timeout,
                         launcher=launcher)

        self.key = key
        self.machine = machine
        self.cores_per_node = cores_per_node
        self.mem_per_node = mem_per_node
        self.exclusive = exclusive
        self.account = account
        self.qos = qos
        self.constraint = constraint

        self.scheduler_options = scheduler_options + '\n'
        if exclusive:
            self.scheduler_options += "#SBATCH --exclusive\n"
        if account:
            self.scheduler_options += "#SBATCH --account={}\n".format(account)
        if qos:
            self.scheduler_options += "#SBATCH --qos={}\n".format(qos)
        if constraint:
            self.scheduler_options += "#SBATCH --constraint={}\n".format(constraint)
        self.worker_init = worker_init + '\n'

        with Client(key=self.key) as client:
            user_info = client.user()
            self.remote_user = user_info.name
            logger.debug(f"sfapi running as remote user {self.remote_user}")
            print(f"sfapi running as remote user {self.remote_user}", flush=True)

        self.workdir = workdir if workdir else f"/global/homes/{self.remote_user[0]}/{self.remote_user}/.remote_parsl"
        print(self.workdir, flush=True)

    def _status(self):
        '''Returns the status list for a list of job_ids

        Args:
              self

        Returns:
              [status...] : Status list of all jobs
        '''
        job_id_list = [jid for jid, job in self.resources.items() if not job['status'].terminal]

        if len(job_id_list) == 0:
            logger.debug('No active jobs, skipping status update')
            return

        with Client(key=self.key) as client:
            logger.debug(f"Calling sfapi on {self.machine}")
            compute = client.compute(self.machine)
            jobs_status = compute.jobs(jobids=job_id_list, command=JobCommand.sacct)

        jobs_missing = set(self.resources.keys())
        for job_status in jobs_status:
            slurm_state = job_status.state
            job_id = job_status.jobid
            if slurm_state not in self._translate_table:
                logger.warning(f"Slurm status {slurm_state} is not recognized")
            status = self._translate_table.get(slurm_state, JobState.UNKNOWN)
            logger.debug("Updating job {} with slurm status {} to parsl state {!s}".format(job_id, slurm_state, status))
            self.resources[job_id]['status'] = JobStatus(status,
                                                         stdout_path=self.resources[job_id]['job_stdout_path'],
                                                         stderr_path=self.resources[job_id]['job_stderr_path'])
            jobs_missing.remove(job_id)

        # sacct can get job info after jobs have completed so this path shouldn't be hit
        # squeue does not report on jobs that are not running. So we are filling in the
        # blanks for missing jobs, we might lose some information about why the jobs failed.
        for missing_job in jobs_missing:
            logger.warning("Updating missing job {} to completed status".format(missing_job))
            self.resources[missing_job]['status'] = JobStatus(
                JobState.COMPLETED, stdout_path=self.resources[missing_job]['job_stdout_path'],
                stderr_path=self.resources[missing_job]['job_stderr_path'])

    def submit(self, command: str, tasks_per_node: int, job_name="parsl.slurm") -> str:
        """Submit the command as a slurm job.

        Parameters
        ----------
        command : str
            Command to be made on the remote side.
        tasks_per_node : int
            Command invocations to be launched per node
        job_name : str
            Name for the job
        Returns
        -------
        job id : str
            A string identifier for the job
        """

        scheduler_options = self.scheduler_options
        worker_init = self.worker_init
        if self.mem_per_node is not None:
            scheduler_options += '#SBATCH --mem={}g\n'.format(self.mem_per_node)
            worker_init += 'export PARSL_MEMORY_GB={}\n'.format(self.mem_per_node)
        if self.cores_per_node is not None:
            cpus_per_task = math.floor(self.cores_per_node / tasks_per_node)
            scheduler_options += '#SBATCH --cpus-per-task={}'.format(cpus_per_task)
            worker_init += 'export PARSL_CORES={}\n'.format(cpus_per_task)

        job_name = "{0}.{1}".format(job_name, time.time())

        assert self.script_dir, "Expected script_dir to be set"
        script_path = os.path.join(self.script_dir, job_name)
        with Client(key=self.key) as client:
            logger.debug(f"Calling sfapi on {self.machine}")
            compute = client.compute(self.machine)
            [remote_script_dir] = compute.ls(self.workdir)
            if not remote_script_dir.is_dir():
                raise SubmitException("Remote directory is not a directory")
        script_path = remote_script_dir
        job_stdout_path = script_path + ".stdout"
        job_stderr_path = script_path + ".stderr"

        logger.debug("Requesting one block with {} nodes".format(self.nodes_per_block))

        job_config: Dict[str, Any] = {}
        job_config["submit_script_dir"] = self.script_dir
        job_config["nodes"] = self.nodes_per_block
        job_config["tasks_per_node"] = tasks_per_node
        job_config["walltime"] = wtime_to_minutes(self.walltime)
        job_config["scheduler_options"] = scheduler_options
        job_config["worker_init"] = worker_init
        job_config["user_script"] = command
        job_config["job_stdout_path"] = job_stdout_path
        job_config["job_stderr_path"] = job_stderr_path

        # Wrap the command
        job_config["user_script"] = self.launcher(command,
                                                  tasks_per_node,
                                                  self.nodes_per_block)

        logger.debug("Writing submit script")
        local_script_dir = Path().cwd() / '.script_path'
        local_script_dir.mkdir(parents=True, exist_ok=True)
        local_script_path = local_script_dir / job_name
        self._write_submit_script(template_string, local_script_path, job_name, job_config)

        with Client(key=self.key) as client:
            logger.debug(f"Calling sfapi on {self.machine}")
            compute = client.compute(self.machine)
            [remote_script_path] = compute.ls(script_path)
            remote_script_path.upload(local_script_path.open('r'))
            job = compute.submit_job(remote_script_path)
            print(job)

        job_id = job.jobid
        status = self._translate_table.get(job.state, JobStatus(JobState.PENDING))
        logger.debug(f"Job: {job_id} submitted with status {status}")
        self.resources[job_id] = {'job_id': job_id,
                                  'status': status,
                                  'job_stdout_path': job_stdout_path,
                                  'job_stderr_path': job_stderr_path,
                                  }
        return job_id

    def cancel(self, job_ids):
        ''' Cancels the jobs specified by a list of job ids

        Args:
        job_ids : [<job_id> ...]

        Returns :
        [True/False...] : If the cancel operation fails the entire list will be False.
        '''

        with Client(key=self.key) as client:
            logger.debug(f"Calling sfapi on {self.machine}")
            compute = client.compute(self.machine)
            jobs = compute.jobs(jobids=job_ids, command=JobCommand.sacct)
            for job in jobs:
                job.cancel(wait=True)
                logger.debug(f"Canceled {job.jobid}")
                self.resources[job.jobid]['status'] = JobStatus(JobState.CANCELLED)

        rets = [True for i in job_ids]

        return rets

    @property
    def status_polling_interval(self):
        return 60
