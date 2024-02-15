#include "types.h"
#include "param.h"
#include "memlayout.h"
#include "riscv.h"
#include "spinlock.h"
#include "proc.h"
#include "defs.h"

struct Chonker{
    uint32 VAblock;
    uint8 valid;
    int pid;
    uint32 LRU;
};
uint64 AVAILABLE_PAGES = (PHYSTOP - KERNBASE) / PGSIZE;
uint64 DONT_MAP = ~((uint64)(0x0));
uint64 PICK_CURRENT_FOR_MAP = ~((uint64)(0x1));
uint64 ACCESS_BIT_MASK = 0x40;

uint64 NO_PROCESS_VICTIM = ~((uint64)0x0);

struct Chonker chonker[4096];
uint8 diskControlBlock[4096] = {0};
int allowYield = 1;

struct cpu cpus[NCPU];

struct proc proc[NPROC];

struct proc *initproc;

int nextpid = 1;
struct spinlock pid_lock;

extern void forkret(void);
static void freeproc(struct proc *p);

extern char trampoline[]; // trampoline.S

// helps ensure that wakeups of wait()ing
// parents are not lost. helps obey the
// memory model when using p->parent.
// must be acquired before any p->lock.
struct spinlock wait_lock;

// Allocate a page for each process's kernel stack.
// Map it high in memory, followed by an invalid
// guard page.
void
proc_mapstacks(pagetable_t kpgtbl)
{
  struct proc *p;
  
  for(p = proc; p < &proc[NPROC]; p++) {
    char *pa = kalloc();
    if(pa == 0)
      panic("kalloc");
    uint64 va = KSTACK((int) (p - proc));
    kvmmap(kpgtbl, va, (uint64)pa, PGSIZE, PTE_R | PTE_W);
  }
}

// initialize the proc table.
void
procinit(void)
{
  struct proc *p;
  
  initlock(&pid_lock, "nextpid");
  initlock(&wait_lock, "wait_lock");
  for(p = proc; p < &proc[NPROC]; p++) {
      initlock(&p->lock, "proc");
      p->state = UNUSED;
      p->kstack = KSTACK((int) (p - proc));
  }
}

// Must be called with interrupts disabled,
// to prevent race with process being moved
// to a different CPU.
int
cpuid()
{
  int id = r_tp();
  return id;
}

// Return this CPU's cpu struct.
// Interrupts must be disabled.
struct cpu*
mycpu(void)
{
  int id = cpuid();
  struct cpu *c = &cpus[id];
  return c;
}

// Return the current struct proc *, or zero if none.
struct proc*
myproc(void)
{
  push_off();
  struct cpu *c = mycpu();
  struct proc *p = c->proc;
  pop_off();
  return p;
}

int
allocpid()
{
  int pid;
  
  acquire(&pid_lock);
  pid = nextpid;
  nextpid = nextpid + 1;
  release(&pid_lock);

  return pid;
}

// Look in the process table for an UNUSED proc.
// If found, initialize state required to run in the kernel,
// and return with p->lock held.
// If there are no free procs, or a memory allocation fails, return 0.
static struct proc*
allocproc(void)
{
  struct proc *p;

  for(p = proc; p < &proc[NPROC]; p++) {
    acquire(&p->lock);
    if(p->state == UNUSED) {
      goto found;
    } else {
      release(&p->lock);
    }
  }
  return 0;

found:
  p->pid = allocpid();
  p->state = USED;

  // Allocate a trapframe page.
  if((p->trapframe = (struct trapframe *)kalloc()) == 0){
    freeproc(p);
    release(&p->lock);
    return 0;
  }

  // An empty user page table.
  p->pagetable = proc_pagetable(p);
  if(p->pagetable == 0){
    freeproc(p);
    release(&p->lock);
    return 0;
  }

  // Set up new context to start executing at forkret,
  // which returns to user space.
  memset(&p->context, 0, sizeof(p->context));
  p->context.ra = (uint64)forkret;
  p->context.sp = p->kstack + PGSIZE;

  return p;
}

// free a proc structure and the data hanging from it,
// including user pages.
// p->lock must be held.
static void
freeproc(struct proc *p)
{
  if(p->trapframe)
    kfree((void*)p->trapframe);
  p->trapframe = 0;
  if(p->pagetable)
    proc_freepagetable(p->pagetable, p->sz);
  p->pagetable = 0;
  p->sz = 0;
  p->pid = 0;
  p->parent = 0;
  p->name[0] = 0;
  p->chan = 0;
  p->killed = 0;
  p->xstate = 0;
  p->state = UNUSED;
}

// Create a user page table for a given process, with no user memory,
// but with trampoline and trapframe pages.
pagetable_t
proc_pagetable(struct proc *p)
{
  pagetable_t pagetable;

  // An empty page table.
  pagetable = uvmcreate();
  if(pagetable == 0)
    return 0;

  // map the trampoline code (for system call return)
  // at the highest user virtual address.
  // only the supervisor uses it, on the way
  // to/from user space, so not PTE_U.
  if(mappages(pagetable, TRAMPOLINE, PGSIZE,
              (uint64)trampoline, PTE_R | PTE_X, 0, 0xFF) < 0){
    uvmfree(pagetable, 0);
    return 0;
  }

  // map the trapframe page just below the trampoline page, for
  // trampoline.S.
  if(mappages(pagetable, TRAPFRAME, PGSIZE,
              (uint64)(p->trapframe), PTE_R | PTE_W, 0,0xFF) < 0){
    uvmunmap(pagetable, TRAMPOLINE, 1, 0);
    uvmfree(pagetable, 0);
    return 0;
  }

  return pagetable;
}

// Free a process's page table, and free the
// physical memory it refers to.
void
proc_freepagetable(pagetable_t pagetable, uint64 sz)
{
  uvmunmap(pagetable, TRAMPOLINE, 1, 0);
  uvmunmap(pagetable, TRAPFRAME, 1, 0);
  uvmfree(pagetable, sz);
}

// a user program that calls exec("/init")
// assembled from ../user/initcode.S
// od -t xC ../user/initcode
uchar initcode[] = {
  0x17, 0x05, 0x00, 0x00, 0x13, 0x05, 0x45, 0x02,
  0x97, 0x05, 0x00, 0x00, 0x93, 0x85, 0x35, 0x02,
  0x93, 0x08, 0x70, 0x00, 0x73, 0x00, 0x00, 0x00,
  0x93, 0x08, 0x20, 0x00, 0x73, 0x00, 0x00, 0x00,
  0xef, 0xf0, 0x9f, 0xff, 0x2f, 0x69, 0x6e, 0x69,
  0x74, 0x00, 0x00, 0x24, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00
};

// Set up first user process.
void
userinit(void)
{
  struct proc *p;

  p = allocproc();
  initproc = p;
  
  // allocate one user page and copy initcode's instructions
  // and data into it.
  uvmfirst(p->pagetable, initcode, sizeof(initcode));
  p->sz = PGSIZE;

  // prepare for the very first "return" from kernel to user.
  p->trapframe->epc = 0;      // user program counter
  p->trapframe->sp = PGSIZE;  // user stack pointer

  safestrcpy(p->name, "initcode", sizeof(p->name));
  p->cwd = namei("/");

  p->state = RUNNABLE;

  release(&p->lock);
}

// Grow or shrink user memory by n bytes.
// Return 0 on success, -1 on failure.
int
growproc(int n) {

    uint64 sz;
    struct proc *p = myproc();

    sz = p->sz;
    if (n > 0) {
        if ((sz = uvmalloc(p->pagetable, sz, sz + n, PTE_W, 1)) == 0) {

            return -1;
        }
    } else if (n < 0) {
        sz = uvmdealloc(p->pagetable, sz, sz + n);
    }
    p->sz = sz;

    return 0;
}

// Create a new process, copying the parent.
// Sets up child kernel stack to return as if from fork() system call.
int
fork(void) {
    noYield();

    int i, pid;
    struct proc *np;
    struct proc *p = myproc();

    // Allocate process.
    if ((np = allocproc()) == 0) {
        yesYield();

        return -1;
    }

    // Copy user memory from parent to child.
    if (uvmcopy(p->pagetable, np->pagetable, p->sz, np->pid) < 0) {
        freeproc(np);
        release(&np->lock);
        yesYield();

        return -1;
    }
    np->sz = p->sz;

    // copy saved user registers.
    *(np->trapframe) = *(p->trapframe);

    // Cause fork to return 0 in the child.
    np->trapframe->a0 = 0;

    // increment reference counts on open file descriptors.
    for (i = 0; i < NOFILE; i++)
        if (p->ofile[i])
            np->ofile[i] = filedup(p->ofile[i]);
    np->cwd = idup(p->cwd);

    safestrcpy(np->name, p->name, sizeof(p->name));

    pid = np->pid;

    release(&np->lock);

    acquire(&wait_lock);
    np->parent = p;
    release(&wait_lock);

    acquire(&np->lock);
    np->state = RUNNABLE;
    release(&np->lock);
    yesYield();

    return pid;
}

// Pass p's abandoned children to init.
// Caller must hold wait_lock.
void
reparent(struct proc *p) {
    struct proc *pp;

    for (pp = proc; pp < &proc[NPROC]; pp++) {
        if (pp->parent == p) {
            pp->parent = initproc;
            wakeup(initproc);
        }
    }
}

// Exit the current process.  Does not return.
// An exited process remains in the zombie state
// until its parent calls wait().
void
exit(int status) {
    struct proc *p = myproc();

    if (p == initproc)
        panic("init exiting");

    // Close all open files.
    for (int fd = 0; fd < NOFILE; fd++) {
        if (p->ofile[fd]) {
            struct file *f = p->ofile[fd];
            fileclose(f);
            p->ofile[fd] = 0;
        }
    }

    begin_op();
    iput(p->cwd);
    end_op();
    p->cwd = 0;

    acquire(&wait_lock);

    // Give any children to init.
    reparent(p);

    // Parent might be sleeping in wait().
    wakeup(p->parent);

    acquire(&p->lock);

    p->xstate = status;
    p->state = ZOMBIE;

    release(&wait_lock);

    // Jump into the scheduler, never to return.
    sched();
    panic("zombie exit");
}

// Wait for a child process to exit and return its pid.
// Return -1 if this process has no children.
int
wait(uint64 addr) {
    struct proc *pp;
    int havekids, pid;
    struct proc *p = myproc();

    acquire(&wait_lock);

    for (;;) {
        // Scan through table looking for exited children.
        havekids = 0;
        for (pp = proc; pp < &proc[NPROC]; pp++) {
            if (pp->parent == p) {
                // make sure the child isn't still in exit() or swtch().
                acquire(&pp->lock);

                havekids = 1;
                if (pp->state == ZOMBIE) {
                    // Found one.
                    pid = pp->pid;
                    if (addr != 0 && copyout(p->pagetable, addr, (char *) &pp->xstate,
                                             sizeof(pp->xstate)) < 0) {
                        release(&pp->lock);
                        release(&wait_lock);
                        return -1;
                    }
                    freeproc(pp);
                    release(&pp->lock);
                    release(&wait_lock);
                    return pid;
                }
                release(&pp->lock);
            }
        }

        // No point waiting if we don't have any children.
        if (!havekids || killed(p)) {
            release(&wait_lock);
            return -1;
        }

        // Wait for a child to exit.
        sleep(p, &wait_lock);  //DOC: wait-sleep
    }
}

// Per-CPU process scheduler.
// Each CPU calls scheduler() after setting itself up.
// Scheduler never returns.  It loops, doing:
//  - choose a process to run.
//  - swtch to start running that process.
//  - eventually that process transfers control
//    via swtch back to the scheduler.
void
scheduler(void) {
    struct proc *p;
    struct cpu *c = mycpu();

    c->proc = 0;
    for (;;) {
        // Avoid deadlock by ensuring that devices can interrupt.
        intr_on();

        for (p = proc; p < &proc[NPROC]; p++) {
            acquire(&p->lock);
            if (p->state == RUNNABLE) {
                // Switch to chosen process.  It is the process's job
                // to release its lock and then reacquire it
                // before jumping back to us.
                p->state = RUNNING;
                c->proc = p;
                swtch(&c->context, &p->context);

                // Process is done running for now.
                // It should have changed its p->state before coming back.
                c->proc = 0;
            }
            release(&p->lock);
        }
    }
}

// Switch to scheduler.  Must hold only p->lock
// and have changed proc->state. Saves and restores
// intena because intena is a property of this
// kernel thread, not this CPU. It should
// be proc->intena and proc->noff, but that would
// break in the few places where a lock is held but
// there's no process.
void
sched(void) {
    int intena;
    struct proc *p = myproc();

    if (!holding(&p->lock))
        panic("sched p->lock");
    if (mycpu()->noff != 1)
        panic("sched locks");
    if (p->state == RUNNING)
        panic("sched running");
    if (intr_get())
        panic("sched interruptible");

    intena = mycpu()->intena;
    swtch(&p->context, &mycpu()->context);
    mycpu()->intena = intena;
}

// Give up the CPU for one scheduling round.
void
yield(void)
{
    if(allowYield <= 0) {
        return;
    }
    struct proc *p = myproc();
    acquire(&p->lock);
    p->state = RUNNABLE;
    sched();
    release(&p->lock);

}

// A fork child's very first scheduling by scheduler()
// will swtch to forkret.
void
forkret(void) {
    static int first = 1;

    // Still holding p->lock from scheduler.
    release(&myproc()->lock);

    if (first) {
        // File system initialization must be run in the context of a
        // regular process (e.g., because it calls sleep), and thus cannot
        // be run from main().
        first = 0;
        fsinit(ROOTDEV);
    }

    usertrapret();
}

// Atomically release lock and sleep on chan.
// Reacquires lock when awakened.
void
sleep(void *chan, struct spinlock *lk) {
    struct proc *p = myproc();

    // Must acquire p->lock in order to
    // change p->state and then call sched.
    // Once we hold p->lock, we can be
    // guaranteed that we won't miss any wakeup
    // (wakeup locks p->lock),
    // so it's okay to release lk.

    acquire(&p->lock);  //DOC: sleeplock1
    release(lk);

    // Go to sleep.
    p->chan = chan;
    p->state = SLEEPING;

    sched();

    // Tidy up.
    p->chan = 0;

    // Reacquire original lock.
    release(&p->lock);
    acquire(lk);
}

// Wake up all processes sleeping on chan.
// Must be called without any p->lock.
void
wakeup(void *chan)
{
  struct proc *p;

  for(p = proc; p < &proc[NPROC]; p++) {
    if(p != myproc()){
      if (holding(&p->lock) && p->state == USED ) {
        // Process is being created.
        continue;
      }
      acquire(&p->lock);
      if(p->state == SLEEPING && p->chan == chan) {
        p->state = RUNNABLE;
      }
      release(&p->lock);
    }
  }
}

// Kill the process with the given pid.
// The victim won't exit until it tries to return
// to user space (see usertrap() in trap.c).
int
kill(int pid)
{
  struct proc *p;

  for(p = proc; p < &proc[NPROC]; p++){
    acquire(&p->lock);
    if(p->pid == pid){
      p->killed = 1;
      if(p->state == SLEEPING){
        // Wake process from sleep().
        p->state = RUNNABLE;
      }
      release(&p->lock);
      return 0;
    }
    release(&p->lock);
  }
  return -1;
}

void
setkilled(struct proc *p)
{
  acquire(&p->lock);
  p->killed = 1;
  release(&p->lock);
}

int
killed(struct proc *p)
{
  int k;
  
  acquire(&p->lock);
  k = p->killed;
  release(&p->lock);
  return k;
}

// Copy to either a user address, or kernel address,
// depending on usr_dst.
// Returns 0 on success, -1 on error.
int
either_copyout(int user_dst, uint64 dst, void *src, uint64 len)
{
  struct proc *p = myproc();
  if(user_dst){
    return copyout(p->pagetable, dst, src, len);
  } else {
    memmove((char *)dst, src, len);
    return 0;
  }
}

// Copy from either a user address, or kernel address,
// depending on usr_src.
// Returns 0 on success, -1 on error.
int
either_copyin(void *dst, int user_src, uint64 src, uint64 len)
{
  struct proc *p = myproc();
  if(user_src){
    return copyin(p->pagetable, dst, src, len);
  } else {
    memmove(dst, (char*)src, len);
    return 0;
  }
}

// Print a process listing to console.  For debugging.
// Runs when user types ^P on console.
// No lock to avoid wedging a stuck machine further.
void
procdump(void)
{
  static char *states[] = {
  [UNUSED]    "unused",
  [USED]      "used",
  [SLEEPING]  "sleep ",
  [RUNNABLE]  "runble",
  [RUNNING]   "run   ",
  [ZOMBIE]    "zombie"
  };
  struct proc *p;
  char *state;

  printf("\n");
  for(p = proc; p < &proc[NPROC]; p++){
    if(p->state == UNUSED)
      continue;
    if(p->state >= 0 && p->state < NELEM(states) && states[p->state])
      state = states[p->state];
    else
      state = "???";
    printf("%d %s %s", p->pid, state, p->name);
    printf("\n");
  }
}

void
initChonker()
{
    allowYield = 1;
    for(int i = 0; i < 4096; i ++) {
        chonker[i].VAblock = 0;
        chonker[i].valid = 0;
        chonker[i].pid = 0;
        chonker[i].LRU = 0;
        diskControlBlock[i] = 0;
    }
}

int
mapBlockRam(uint64 PA, uint64 VA, uint64 pid)
{
    int entry = (PA - KERNBASE) >> 12; // set entry
    if(entry == 3961){
        int c = 3;
        c++;
    }
    if(chonker[entry].valid != 0) { // error, entry already taken
        return -1;
    }
    if(pid == DONT_MAP){
        return 1;
    }
    struct proc* p = myproc();
    if(pid == PICK_CURRENT_FOR_MAP){
        if(p == 0){
            return -3;
        }
        pid = myproc()->pid;
    }
    // set pid
    chonker[entry].pid = pid;

    // set valid bit
    chonker[entry].valid = 1;

    // set VA
    chonker[entry].VAblock = VA >> 12;

    // set LRU
    chonker[entry].LRU = 0;

    return 1;
}

void
unmapBlockRam(uint64 PA)
{
    int entry = (PA - KERNBASE) >> 12;
    chonker[entry].valid = 0;
    chonker[entry].pid = 0;
    chonker[entry].VAblock = 0;
    chonker[entry].LRU = 0;
}

int
findFreeBlockOnDisk()
{
    for(int i = 0; i < PGSIZE; i ++){
        if(diskControlBlock[i] == 0) {
            return i;
        }
    }
    return PGSIZE;
}

void
noYield()
{
    allowYield--;
}

void
yesYield()
{
    allowYield++;
}

int
swapOut()
{
    // all the race condition risky-hazardous memory structures are protected in kalloc with a mem lock
    // So we have to make sure that swapOut is called only in that kalloc within the critical section


    // find page for swapping
    uint64 entry = PGSIZE;
    uint32 LRUmin = 0xFFFFFFFF;
    uint32 LRUtemp = 0;

    for(int i = 0; i < PGSIZE; i++)
    {
        if( chonker[i].valid != 0){
            LRUtemp = chonker[i].LRU;
            if(LRUtemp <= LRUmin){
                LRUmin = LRUtemp;
                entry = i;
            }
        }
    }

    // no pages for swapping
    if(entry == PGSIZE){
        return -1;
    }

    int pid = chonker[entry].pid;
    struct proc* p = &proc[findProcStructure(pid)];
    uint64 PA = KERNBASE + (entry << 12);
    uint64 VA = chonker[entry].VAblock << 12;

    // check if physical address and pte physical address match
//    uint64 check = walkaddr(p->pagetable, VA);
//    if(PA != check) {
//        return -2; // fatal error
//    }

    // page descriptor
    pte_t* pte = walk(p->pagetable, VA, 0);
    if(pte == 0){
        return -3;
    }

    uint64 blockNo = (uint64)findFreeBlockOnDisk(); // find free block on disk
    // no free block
    if(blockNo == PGSIZE){
        return -4;
    }

    noYield(); // forbid yield
    write_block(blockNo, (uint8*)PA, 1); // swap
    yesYield(); // allow yield

    // devalidate entry in chonker
    chonker[entry].valid = 0;
    chonker[entry].pid = 0;
    chonker[entry].VAblock = 0;
    chonker[entry].LRU = 0;

    diskControlBlock[blockNo] = 0xFF; // set block on disk as taken (7th bit is valid)

    *pte &= 0xFFC00000000003FE; // set valid to 0 and clear virtual physical address
    *pte |= 0x100; // set bit 7 to 1 to indicate that page is on the disk
    *pte |= blockNo << 10; // set page location on disk in descriptor

    kfree((void*)PA);

    return 1;
}

int
swapIn(uint64 VA)
{
    struct proc* p = myproc();

    pte_t* pte = walk(p->pagetable, VA, 0);

    // descriptor non existant
    if(pte == 0){
        return -1;
    }

    uint64 validBit = (*pte) & 0x0000000000000001;

    // no need for swap in, error
    if(validBit == 1){
        return -2;
    }

    uint64 blockNo = ((*pte) & 0xFFC00) >> 10;

    // block is not on the disk
    if(diskControlBlock[blockNo] == 0){
        return -3;
    }

//    // block is on the disk, but it doesn't belong to this process
//    if( (diskControlBlock[blockNo] & ) != (uint8)(p->pid) ){
//        return -4;
//    }

    uint8* mem = kalloc();

    // failure to allocate a block in RAM
    if(mem == 0){
        return -5;
    }

    noYield();
    read_block(blockNo, mem, 1);
    yesYield();

    // set descritptor
    (*pte) &= 0xFFC00000000003FF; // clear blockNo for disk in pte
    (*pte) |= ((uint64)mem >> 12) << 10;// insert address
    (*pte) |= (uint64)1; // set valid bit to true

    // set diskControlBlock
    diskControlBlock[blockNo] = 0;

    mapBlockRam((uint64)mem, VA, PICK_CURRENT_FOR_MAP);

    return 1;
}



int
checkMapPermissionProc(uint64 VA)
{
    struct proc* p = myproc();
    if(p==0){
        return 0;
    }

    uint64 PA = walkaddr(p->pagetable, VA);
    if(PA == 0){
        return 0;
    }
    int entry = (PA - KERNBASE) >> 12;

    if( chonker[entry].valid == 0){
        return 0;
    }
    if(chonker[entry].pid != p->pid){
        return 0;
    }

    return 1;
}
int
findProcStructure(int pid)
{
    for(int i = 0; i < NPROC; i ++){
        if(proc[i].state == UNUSED){
            continue;
        }
        if(proc[i].pid == pid){
            return i;
        }
    }
    panic("NO PROC");
}

int
updateLRU() {
    struct proc *p;
    uint64 VA;
    uint64 pid;
    pte_t *pte;
    uint64 accessBit;
  //  uint64 PA;
    uint64 entry;
    noYield();

    for (int i = 0; i < PGSIZE; i++) {
        // check if the entry is valid
        if (chonker[i].valid == 0) {
            continue;
        }

        // get Virtual address of the block
        VA = (uint64)chonker[i].VAblock << 12;
        if (VA == 0) {
            yesYield();
            return -1;
        }

        pid = chonker[i].pid;
        entry = findProcStructure(pid);

        p = &proc[entry];
        if (p->state == ZOMBIE) {
            continue;
        }

        pte = walk(p->pagetable, VA, 0);
        if (pte == 0) {
            yesYield();
            return -2;
        }
        accessBit = (*pte & ACCESS_BIT_MASK) << 25;

        chonker[i].LRU = (chonker[i].LRU >> 1) | accessBit;

    }
//    for(int i = 0; i < NPROC; i ++){
//        if(proc[i].state == UNUSED || proc[i].state == ZOMBIE){
//            continue;
//        }
//        for(int j = 0; j < proc[i].sz; j += PGSIZE){
//            PA = walkaddr(proc[i].pagetable, j);
//            if(PA == 0){
//                continue;
//            }
//            if(i==3){
//                int c = 0;
//                c++;
//            }
//            pte = walk(proc[i].pagetable,j,0);
//            accessBit = (*pte & ACCESS_BIT) << 25;
//            entry = (PA - KERNBASE) >> 12;
//            if((chonker[entry].validAndPid & VALID_BIT_MASK) == 0){
//               // panic("CHONKER NOT ALLOC");
//            }
//            if((chonker[entry].valdiAndPid & PID_MASK) != proc[i].pid){
//               // panic("NOT SAME PID");
//            }
//            chonker[i].LRU = (chonker[i].LRU >> 1) | accessBit;
//        }
//    }
    thrashing();
    yesYield();
    return 1;
}

void
freeFromDisk(uint64 pte)
{
    int entry = (pte & 0xFFC00) >> 10;
    diskControlBlock[entry] = 0;
}

int
calculateReferences(int entry)
{
    pte_t* pte;
    uint64 accessBit;
    int counter = 0;
    for(int i = 0; i < proc[entry].sz; i += PGSIZE){
        pte = walk(proc[entry].pagetable, i, 0);
        accessBit = *pte & ACCESS_BIT_MASK;
        if(accessBit != 0){
            counter++;
        }
    }
    return counter;
}

uint64
findProcessVictim()
{
    struct proc* p = myproc();
    if(p == 0){
        return NO_PROCESS_VICTIM;
    }
    int pid = p->pid;

    for(int i = 0; i < NPROC; i ++){
        if(proc[i].pid == 0){
            continue;
        }
        if(proc[i].pid == 1){
            continue;
        }
        if(proc[i].pid == pid){
            continue;
        }
        if(proc[i].state == RUNNABLE){
            return (uint64)i;
        }
    }

    return NO_PROCESS_VICTIM;
}

void
rethrash()
{
    for(int i = 0; i < NPROC; i ++){
        if(proc[i].state == THRASHED){
            proc[i].state = RUNNABLE;
        }
    }
}

void
thrashing()
{
    uint64 workingSet = 0;
    for(int i = 0; i < NPROC; i ++){
        if(proc[i].state == UNUSED){
            continue;
        }
        if(proc[i].state == ZOMBIE){
            continue;
        }
        workingSet += (uint64)calculateReferences(i);
    }

    uint64 entry;
    // Check if thrashing is occurring
    if(workingSet > AVAILABLE_PAGES){
        entry = findProcessVictim();
        if(entry == NO_PROCESS_VICTIM){
            return;
        }
        proc[entry].state = THRASHED;
    }
    else{ // put thrashed processes back in game
        rethrash();
    }
}