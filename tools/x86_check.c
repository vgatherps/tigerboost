#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern double predict(const double *row);

static char *join_path(const char *directory, const char *name) {
    size_t directory_length = strlen(directory);
    size_t name_length = strlen(name);
    if (directory_length > SIZE_MAX - name_length - 2) {
        return NULL;
    }
    char *path = malloc(directory_length + name_length + 2);
    if (path == NULL) {
        return NULL;
    }
    snprintf(path, directory_length + name_length + 2, "%s/%s", directory, name);
    return path;
}

static int parse_count(const char *line, const char *key, size_t *value) {
    size_t key_length = strlen(key);
    if (strncmp(line, key, key_length) != 0 || line[key_length] != ' ') {
        return 0;
    }
    const char *number = line + key_length + 1;
    if (*number == '\0' || *number == '-' || *number == '+') {
        return -1;
    }
    errno = 0;
    char *end = NULL;
    uintmax_t parsed = strtoumax(number, &end, 10);
    if (errno != 0 || end == number || (*end != '\n' && *end != '\0') || parsed == 0 ||
        parsed > SIZE_MAX) {
        return -1;
    }
    *value = (size_t)parsed;
    return 1;
}

static int read_manifest(const char *directory, size_t *features, size_t *rows) {
    char *path = join_path(directory, "manifest.txt");
    if (path == NULL) {
        fprintf(stderr, "manifest path is too long\n");
        return 0;
    }
    FILE *file = fopen(path, "r");
    if (file == NULL) {
        fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno));
        free(path);
        return 0;
    }
    char line[512];
    int have_features = 0;
    int have_rows = 0;
    int have_avx2 = 0;
    int variants = 0;
    int valid = 1;
    while (fgets(line, sizeof(line), file) != NULL) {
        if (strchr(line, '\n') == NULL && !feof(file)) {
            valid = 0;
            break;
        }
        int parsed = parse_count(line, "features", features);
        if (parsed != 0) {
            if (parsed < 0 || have_features) {
                valid = 0;
                break;
            }
            have_features = 1;
            continue;
        }
        parsed = parse_count(line, "rows", rows);
        if (parsed != 0) {
            if (parsed < 0 || have_rows) {
                valid = 0;
                break;
            }
            have_rows = 1;
            continue;
        }
        if (strncmp(line, "avx2 ", 5) == 0) {
            if (have_avx2 || (strcmp(line, "avx2 0\n") != 0 && strcmp(line, "avx2 1\n") != 0 &&
                             strcmp(line, "avx2 0") != 0 && strcmp(line, "avx2 1") != 0)) {
                valid = 0;
                break;
            }
            have_avx2 = 1;
            continue;
        }
        if (strncmp(line, "variant ", 8) == 0 && line[8] != '\0' && line[8] != '\n') {
            char *end = strchr(line + 8, '\n');
            if (end != NULL) {
                *end = '\0';
            }
            if (strchr(line + 8, ' ') != NULL || strchr(line + 8, '\t') != NULL) {
                valid = 0;
                break;
            }
            variants++;
            continue;
        }
        valid = 0;
        break;
    }
    if (ferror(file)) {
        fprintf(stderr, "cannot read %s: %s\n", path, strerror(errno));
        valid = 0;
    }
    if (fclose(file) != 0) {
        fprintf(stderr, "cannot close %s: %s\n", path, strerror(errno));
        valid = 0;
    }
    if (!valid || !have_features || !have_rows || variants == 0) {
        fprintf(stderr, "mismatched manifest %s\n", path);
        free(path);
        return 0;
    }
    free(path);
    return 1;
}

static int read_exact(const char *directory, const char *name, void *data, size_t bytes) {
    char *path = join_path(directory, name);
    if (path == NULL) {
        fprintf(stderr, "%s path is too long\n", name);
        return 0;
    }
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno));
        free(path);
        return 0;
    }
    size_t read = fread(data, 1, bytes, file);
    int extra = fgetc(file);
    int valid = read == bytes && extra == EOF && !ferror(file);
    if (!valid) {
        fprintf(stderr, "%s has %zu bytes, expected %zu\n", path, read, bytes);
    }
    if (fclose(file) != 0) {
        fprintf(stderr, "cannot close %s: %s\n", path, strerror(errno));
        valid = 0;
    }
    free(path);
    return valid;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: x86_check <dir>\n");
        return 2;
    }
    size_t features = 0;
    size_t rows = 0;
    if (!read_manifest(argv[1], &features, &rows)) {
        return 2;
    }
    if (features > (SIZE_MAX - 63) / sizeof(double) || rows > SIZE_MAX / sizeof(double) ||
        rows > SIZE_MAX / features / sizeof(double)) {
        fprintf(stderr, "manifest dimensions exceed addressable storage\n");
        return 2;
    }
    size_t row_bytes = features * sizeof(double);
    size_t rows_bytes = rows * row_bytes;
    size_t expected_bytes = rows * sizeof(double);
    double *input = malloc(rows_bytes);
    double *expected = malloc(expected_bytes);
    size_t aligned_bytes = (row_bytes + 63) & ~(size_t)63;
    double *aligned_row = aligned_alloc(64, aligned_bytes);
    if (input == NULL || expected == NULL || aligned_row == NULL) {
        fprintf(stderr, "cannot allocate input buffers\n");
        free(input);
        free(expected);
        free(aligned_row);
        return 2;
    }
    if (!read_exact(argv[1], "rows.bin", input, rows_bytes) ||
        !read_exact(argv[1], "expected.bin", expected, expected_bytes)) {
        free(input);
        free(expected);
        free(aligned_row);
        return 2;
    }
    for (size_t row = 0; row < rows; row++) {
        memcpy(aligned_row, input + row * features, row_bytes);
        double actual = predict(aligned_row);
        if (memcmp(&actual, &expected[row], sizeof(double)) != 0) {
            uint64_t actual_bits;
            uint64_t expected_bits;
            memcpy(&actual_bits, &actual, sizeof(actual_bits));
            memcpy(&expected_bits, &expected[row], sizeof(expected_bits));
            fprintf(stderr, "row %zu: got %016" PRIx64 " expected %016" PRIx64 "\n", row,
                    actual_bits, expected_bits);
            free(input);
            free(expected);
            free(aligned_row);
            return 1;
        }
    }
    printf("ok %zu rows\n", rows);
    free(input);
    free(expected);
    free(aligned_row);
    return 0;
}
