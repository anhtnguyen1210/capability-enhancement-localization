#define _GNU_SOURCE
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/landlock.h>
#include <linux/seccomp.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <fcntl.h>

#ifndef LANDLOCK_ACCESS_FS_REFER
#define LANDLOCK_ACCESS_FS_REFER (1ULL << 13)
#endif
#ifndef LANDLOCK_ACCESS_FS_TRUNCATE
#define LANDLOCK_ACCESS_FS_TRUNCATE (1ULL << 14)
#endif

static void install_no_network_filter(void) {
    struct sock_filter filter[] = {
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, arch)),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_X86_64, 1, 0),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_socket, 0, 9),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, args[0])),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AF_INET, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AF_INET6, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AF_NETLINK, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AF_PACKET, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
    };
    struct sock_fprog program = {
        .len = (unsigned short)(sizeof(filter) / sizeof(filter[0])),
        .filter = filter,
    };
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 ||
        prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program) != 0) {
        perror("install seccomp filter");
        exit(125);
    }
}

#define DENY_SYSCALL(name) \
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_##name, 0, 1), \
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM)

static void install_no_filesystem_writes_filter(void) {
    const unsigned int write_flags = O_WRONLY | O_RDWR | O_CREAT | O_TRUNC | O_APPEND;
    struct sock_filter filter[] = {
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, arch)),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_X86_64, 1, 0),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
        DENY_SYSCALL(unlink), DENY_SYSCALL(unlinkat),
        DENY_SYSCALL(rename), DENY_SYSCALL(renameat), DENY_SYSCALL(renameat2),
        DENY_SYSCALL(mkdir), DENY_SYSCALL(mkdirat), DENY_SYSCALL(rmdir),
        DENY_SYSCALL(link), DENY_SYSCALL(linkat), DENY_SYSCALL(symlink), DENY_SYSCALL(symlinkat),
        DENY_SYSCALL(chmod), DENY_SYSCALL(fchmod), DENY_SYSCALL(fchmodat),
        DENY_SYSCALL(chown), DENY_SYSCALL(fchown), DENY_SYSCALL(fchownat), DENY_SYSCALL(lchown),
        DENY_SYSCALL(truncate), DENY_SYSCALL(ftruncate),
        DENY_SYSCALL(mknod), DENY_SYSCALL(mknodat),
        DENY_SYSCALL(mount), DENY_SYSCALL(umount2), DENY_SYSCALL(pivot_root),
        DENY_SYSCALL(setxattr), DENY_SYSCALL(lsetxattr), DENY_SYSCALL(fsetxattr),
        DENY_SYSCALL(removexattr), DENY_SYSCALL(lremovexattr), DENY_SYSCALL(fremovexattr),
        DENY_SYSCALL(utime), DENY_SYSCALL(utimes), DENY_SYSCALL(futimesat), DENY_SYSCALL(utimensat),
        DENY_SYSCALL(open_by_handle_at), DENY_SYSCALL(io_uring_setup),
        DENY_SYSCALL(copy_file_range),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_open, 0, 4),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, args[1])),
        BPF_STMT(BPF_ALU | BPF_AND | BPF_K, write_flags),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, 0, 1, 0),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_openat, 0, 4),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, args[2])),
        BPF_STMT(BPF_ALU | BPF_AND | BPF_K, write_flags),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, 0, 1, 0),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
#ifdef SYS_openat2
        DENY_SYSCALL(openat2),
#endif
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
    };
    struct sock_fprog program = {
        .len = (unsigned short)(sizeof(filter) / sizeof(filter[0])),
        .filter = filter,
    };
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 ||
        prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program) != 0) {
        perror("install filesystem seccomp filter");
        exit(125);
    }
}

static void set_limit(int resource, rlim_t value, const char *name) {
    struct rlimit limit = {.rlim_cur = value, .rlim_max = value};
    if (setrlimit(resource, &limit) != 0) {
        perror(name);
        exit(125);
    }
}

static void add_landlock_path(int ruleset_fd, const char *path, uint64_t access) {
    int path_fd = open(path, O_PATH | O_CLOEXEC);
    if (path_fd < 0) {
        perror(path);
        exit(125);
    }
    struct landlock_path_beneath_attr rule = {
        .allowed_access = access,
        .parent_fd = path_fd,
    };
    if (syscall(__NR_landlock_add_rule, ruleset_fd, LANDLOCK_RULE_PATH_BENEATH, &rule, 0) != 0) {
        perror("landlock_add_rule");
        close(path_fd);
        exit(125);
    }
    close(path_fd);
}

static void install_landlock(const char *writable_dir) {
    int abi = syscall(__NR_landlock_create_ruleset, NULL, 0, LANDLOCK_CREATE_RULESET_VERSION);
    if (abi < 1) {
        perror("landlock ABI unavailable");
        exit(125);
    }
    uint64_t read_access = LANDLOCK_ACCESS_FS_EXECUTE | LANDLOCK_ACCESS_FS_READ_FILE |
        LANDLOCK_ACCESS_FS_READ_DIR;
    uint64_t write_access = LANDLOCK_ACCESS_FS_WRITE_FILE | LANDLOCK_ACCESS_FS_REMOVE_DIR |
        LANDLOCK_ACCESS_FS_REMOVE_FILE | LANDLOCK_ACCESS_FS_MAKE_CHAR |
        LANDLOCK_ACCESS_FS_MAKE_DIR | LANDLOCK_ACCESS_FS_MAKE_REG |
        LANDLOCK_ACCESS_FS_MAKE_SOCK | LANDLOCK_ACCESS_FS_MAKE_FIFO |
        LANDLOCK_ACCESS_FS_MAKE_BLOCK | LANDLOCK_ACCESS_FS_MAKE_SYM;
    if (abi >= 2) write_access |= LANDLOCK_ACCESS_FS_REFER;
    if (abi >= 3) write_access |= LANDLOCK_ACCESS_FS_TRUNCATE;
    uint64_t handled = read_access | write_access;
    struct landlock_ruleset_attr ruleset = {.handled_access_fs = handled};
    int ruleset_fd = syscall(__NR_landlock_create_ruleset, &ruleset, sizeof(ruleset), 0);
    if (ruleset_fd < 0) {
        perror("landlock_create_ruleset");
        exit(125);
    }
    add_landlock_path(ruleset_fd, writable_dir, handled);
    add_landlock_path(ruleset_fd, "/usr", LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_READ_DIR);
    add_landlock_path(ruleset_fd, "/lib", LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_READ_DIR);
    add_landlock_path(ruleset_fd, "/lib64", LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_READ_DIR);
    add_landlock_path(ruleset_fd, "/etc", LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_READ_DIR);
    add_landlock_path(ruleset_fd, "/dev/null", LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_WRITE_FILE);
    add_landlock_path(ruleset_fd, "/dev/urandom", LANDLOCK_ACCESS_FS_READ_FILE);
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 ||
        syscall(__NR_landlock_restrict_self, ruleset_fd, 0) != 0) {
        perror("landlock_restrict_self");
        close(ruleset_fd);
        exit(125);
    }
    close(ruleset_fd);
}

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: humaneval_seccomp_exec MEMORY_BYTES PIDS CPU_SECONDS FILE_BYTES LANDLOCK_DIR_OR_DASH COMMAND [ARG...]\n");
        return 125;
    }
    set_limit(RLIMIT_AS, strtoull(argv[1], NULL, 10), "RLIMIT_AS");
    set_limit(RLIMIT_NPROC, strtoull(argv[2], NULL, 10), "RLIMIT_NPROC");
    set_limit(RLIMIT_CPU, strtoull(argv[3], NULL, 10), "RLIMIT_CPU");
    set_limit(RLIMIT_FSIZE, strtoull(argv[4], NULL, 10), "RLIMIT_FSIZE");
    if (strncmp(argv[5], "landlock:", 9) == 0) install_landlock(argv[5] + 9);
    if (strcmp(argv[5], "deny-writes") == 0) install_no_filesystem_writes_filter();
    install_no_network_filter();
    execvp(argv[6], &argv[6]);
    perror("execvp");
    return 125;
}
